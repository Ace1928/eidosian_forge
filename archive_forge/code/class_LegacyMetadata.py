from __future__ import unicode_literals
import codecs
from email import message_from_file
import json
import logging
import re
from . import DistlibException, __version__
from .compat import StringIO, string_types, text_type
from .markers import interpret
from .util import extract_by_key, get_extras
from .version import get_scheme, PEP440_VERSION_RE
class LegacyMetadata(object):
    """The legacy metadata of a release.

    Supports versions 1.0, 1.1, 1.2, 2.0 and 1.3/2.1 (auto-detected). You can
    instantiate the class with one of these arguments (or none):
    - *path*, the path to a metadata file
    - *fileobj* give a file-like object with metadata as content
    - *mapping* is a dict-like object
    - *scheme* is a version scheme name
    """

    def __init__(self, path=None, fileobj=None, mapping=None, scheme='default'):
        if [path, fileobj, mapping].count(None) < 2:
            raise TypeError('path, fileobj and mapping are exclusive')
        self._fields = {}
        self.requires_files = []
        self._dependencies = None
        self.scheme = scheme
        if path is not None:
            self.read(path)
        elif fileobj is not None:
            self.read_file(fileobj)
        elif mapping is not None:
            self.update(mapping)
            self.set_metadata_version()

    def set_metadata_version(self):
        self._fields['Metadata-Version'] = _best_version(self._fields)

    def _write_field(self, fileobj, name, value):
        fileobj.write('%s: %s\n' % (name, value))

    def __getitem__(self, name):
        return self.get(name)

    def __setitem__(self, name, value):
        return self.set(name, value)

    def __delitem__(self, name):
        field_name = self._convert_name(name)
        try:
            del self._fields[field_name]
        except KeyError:
            raise KeyError(name)

    def __contains__(self, name):
        return name in self._fields or self._convert_name(name) in self._fields

    def _convert_name(self, name):
        if name in _ALL_FIELDS:
            return name
        name = name.replace('-', '_').lower()
        return _ATTR2FIELD.get(name, name)

    def _default_value(self, name):
        if name in _LISTFIELDS or name in _ELEMENTSFIELD:
            return []
        return 'UNKNOWN'

    def _remove_line_prefix(self, value):
        if self.metadata_version in ('1.0', '1.1'):
            return _LINE_PREFIX_PRE_1_2.sub('\n', value)
        else:
            return _LINE_PREFIX_1_2.sub('\n', value)

    def __getattr__(self, name):
        if name in _ATTR2FIELD:
            return self[name]
        raise AttributeError(name)

    def get_fullname(self, filesafe=False):
        """Return the distribution name with version.

        If filesafe is true, return a filename-escaped form."""
        return _get_name_and_version(self['Name'], self['Version'], filesafe)

    def is_field(self, name):
        """return True if name is a valid metadata key"""
        name = self._convert_name(name)
        return name in _ALL_FIELDS

    def is_multi_field(self, name):
        name = self._convert_name(name)
        return name in _LISTFIELDS

    def read(self, filepath):
        """Read the metadata values from a file path."""
        fp = codecs.open(filepath, 'r', encoding='utf-8')
        try:
            self.read_file(fp)
        finally:
            fp.close()

    def read_file(self, fileob):
        """Read the metadata values from a file object."""
        msg = message_from_file(fileob)
        self._fields['Metadata-Version'] = msg['metadata-version']
        for field in _ALL_FIELDS:
            if field not in msg:
                continue
            if field in _LISTFIELDS:
                values = msg.get_all(field)
                if field in _LISTTUPLEFIELDS and values is not None:
                    values = [tuple(value.split(',')) for value in values]
                self.set(field, values)
            else:
                value = msg[field]
                if value is not None and value != 'UNKNOWN':
                    self.set(field, value)
        body = msg.get_payload()
        self['Description'] = body if body else self['Description']

    def write(self, filepath, skip_unknown=False):
        """Write the metadata fields to filepath."""
        fp = codecs.open(filepath, 'w', encoding='utf-8')
        try:
            self.write_file(fp, skip_unknown)
        finally:
            fp.close()

    def write_file(self, fileobject, skip_unknown=False):
        """Write the PKG-INFO format data to a file object."""
        self.set_metadata_version()
        for field in _version2fieldlist(self['Metadata-Version']):
            values = self.get(field)
            if skip_unknown and values in ('UNKNOWN', [], ['UNKNOWN']):
                continue
            if field in _ELEMENTSFIELD:
                self._write_field(fileobject, field, ','.join(values))
                continue
            if field not in _LISTFIELDS:
                if field == 'Description':
                    if self.metadata_version in ('1.0', '1.1'):
                        values = values.replace('\n', '\n        ')
                    else:
                        values = values.replace('\n', '\n       |')
                values = [values]
            if field in _LISTTUPLEFIELDS:
                values = [','.join(value) for value in values]
            for value in values:
                self._write_field(fileobject, field, value)

    def update(self, other=None, **kwargs):
        """Set metadata values from the given iterable `other` and kwargs.

        Behavior is like `dict.update`: If `other` has a ``keys`` method,
        they are looped over and ``self[key]`` is assigned ``other[key]``.
        Else, ``other`` is an iterable of ``(key, value)`` iterables.

        Keys that don't match a metadata field or that have an empty value are
        dropped.
        """

        def _set(key, value):
            if key in _ATTR2FIELD and value:
                self.set(self._convert_name(key), value)
        if not other:
            pass
        elif hasattr(other, 'keys'):
            for k in other.keys():
                _set(k, other[k])
        else:
            for k, v in other:
                _set(k, v)
        if kwargs:
            for k, v in kwargs.items():
                _set(k, v)

    def set(self, name, value):
        """Control then set a metadata field."""
        name = self._convert_name(name)
        if (name in _ELEMENTSFIELD or name == 'Platform') and (not isinstance(value, (list, tuple))):
            if isinstance(value, string_types):
                value = [v.strip() for v in value.split(',')]
            else:
                value = []
        elif name in _LISTFIELDS and (not isinstance(value, (list, tuple))):
            if isinstance(value, string_types):
                value = [value]
            else:
                value = []
        if logger.isEnabledFor(logging.WARNING):
            project_name = self['Name']
            scheme = get_scheme(self.scheme)
            if name in _PREDICATE_FIELDS and value is not None:
                for v in value:
                    if not scheme.is_valid_matcher(v.split(';')[0]):
                        logger.warning("'%s': '%s' is not valid (field '%s')", project_name, v, name)
            elif name in _VERSIONS_FIELDS and value is not None:
                if not scheme.is_valid_constraint_list(value):
                    logger.warning("'%s': '%s' is not a valid version (field '%s')", project_name, value, name)
            elif name in _VERSION_FIELDS and value is not None:
                if not scheme.is_valid_version(value):
                    logger.warning("'%s': '%s' is not a valid version (field '%s')", project_name, value, name)
        if name in _UNICODEFIELDS:
            if name == 'Description':
                value = self._remove_line_prefix(value)
        self._fields[name] = value

    def get(self, name, default=_MISSING):
        """Get a metadata field."""
        name = self._convert_name(name)
        if name not in self._fields:
            if default is _MISSING:
                default = self._default_value(name)
            return default
        if name in _UNICODEFIELDS:
            value = self._fields[name]
            return value
        elif name in _LISTFIELDS:
            value = self._fields[name]
            if value is None:
                return []
            res = []
            for val in value:
                if name not in _LISTTUPLEFIELDS:
                    res.append(val)
                else:
                    res.append((val[0], val[1]))
            return res
        elif name in _ELEMENTSFIELD:
            value = self._fields[name]
            if isinstance(value, string_types):
                return value.split(',')
        return self._fields[name]

    def check(self, strict=False):
        """Check if the metadata is compliant. If strict is True then raise if
        no Name or Version are provided"""
        self.set_metadata_version()
        missing, warnings = ([], [])
        for attr in ('Name', 'Version'):
            if attr not in self:
                missing.append(attr)
        if strict and missing != []:
            msg = 'missing required metadata: %s' % ', '.join(missing)
            raise MetadataMissingError(msg)
        for attr in ('Home-page', 'Author'):
            if attr not in self:
                missing.append(attr)
        if self['Metadata-Version'] != '1.2':
            return (missing, warnings)
        scheme = get_scheme(self.scheme)

        def are_valid_constraints(value):
            for v in value:
                if not scheme.is_valid_matcher(v.split(';')[0]):
                    return False
            return True
        for fields, controller in ((_PREDICATE_FIELDS, are_valid_constraints), (_VERSIONS_FIELDS, scheme.is_valid_constraint_list), (_VERSION_FIELDS, scheme.is_valid_version)):
            for field in fields:
                value = self.get(field, None)
                if value is not None and (not controller(value)):
                    warnings.append("Wrong value for '%s': %s" % (field, value))
        return (missing, warnings)

    def todict(self, skip_missing=False):
        """Return fields as a dict.

        Field names will be converted to use the underscore-lowercase style
        instead of hyphen-mixed case (i.e. home_page instead of Home-page).
        This is as per https://www.python.org/dev/peps/pep-0566/#id17.
        """
        self.set_metadata_version()
        fields = _version2fieldlist(self['Metadata-Version'])
        data = {}
        for field_name in fields:
            if not skip_missing or field_name in self._fields:
                key = _FIELD2ATTR[field_name]
                if key != 'project_url':
                    data[key] = self[field_name]
                else:
                    data[key] = [','.join(u) for u in self[field_name]]
        return data

    def add_requirements(self, requirements):
        if self['Metadata-Version'] == '1.1':
            for field in ('Obsoletes', 'Requires', 'Provides'):
                if field in self:
                    del self[field]
        self['Requires-Dist'] += requirements

    def keys(self):
        return list(_version2fieldlist(self['Metadata-Version']))

    def __iter__(self):
        for key in self.keys():
            yield key

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def __repr__(self):
        return '<%s %s %s>' % (self.__class__.__name__, self.name, self.version)