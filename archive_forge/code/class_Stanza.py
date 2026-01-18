import re
from .. import osutils
from ..iterablefile import IterableFile
class Stanza:
    """One stanza for rio.

    Each stanza contains a set of named fields.

    Names must be non-empty ascii alphanumeric plus _.  Names can be repeated
    within a stanza.  Names are case-sensitive.  The ordering of fields is
    preserved.

    Each field value must be either an int or a string.
    """
    __slots__ = ['items']

    def __init__(self, **kwargs):
        """Construct a new Stanza.

        The keyword arguments, if any, are added in sorted order to the stanza.
        """
        self.items = []
        if kwargs:
            for tag, value in sorted(kwargs.items()):
                self.add(tag, value)

    def add(self, tag, value):
        """Append a name and value to the stanza."""
        if not valid_tag(tag):
            raise ValueError('invalid tag {!r}'.format(tag))
        if isinstance(value, bytes):
            pass
        elif isinstance(value, str):
            pass
        elif isinstance(value, Stanza):
            pass
        else:
            raise TypeError('invalid type for rio value: %r of type %s' % (value, type(value)))
        self.items.append((tag, value))

    @classmethod
    def from_pairs(cls, pairs):
        ret = cls()
        ret.items = pairs
        return ret

    def __contains__(self, find_tag):
        """True if there is any field in this stanza with the given tag."""
        for tag, value in self.items:
            if tag == find_tag:
                return True
        return False

    def __len__(self):
        """Return number of pairs in the stanza."""
        return len(self.items)

    def __eq__(self, other):
        if not isinstance(other, Stanza):
            return False
        return self.items == other.items

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'Stanza(%r)' % self.items

    def iter_pairs(self):
        """Return iterator of tag, value pairs."""
        return iter(self.items)

    def to_lines(self):
        """Generate sequence of lines for external version of this file.

        The lines are always utf-8 encoded strings.
        """
        if not self.items:
            return []
        result = []
        for text_tag, text_value in self.items:
            tag = text_tag.encode('ascii')
            if isinstance(text_value, str):
                value = text_value.encode('utf-8', 'surrogateescape')
            elif isinstance(text_value, Stanza):
                value = text_value.to_string()
            else:
                value = text_value
            if value == b'':
                result.append(tag + b': \n')
            elif b'\n' in value:
                val_lines = value.split(b'\n')
                result.append(tag + b': ' + val_lines[0] + b'\n')
                for line in val_lines[1:]:
                    result.append(b'\t' + line + b'\n')
            else:
                result.append(tag + b': ' + value + b'\n')
        return result

    def to_string(self):
        """Return stanza as a single string"""
        return b''.join(self.to_lines())

    def write(self, to_file):
        """Write stanza to a file"""
        to_file.writelines(self.to_lines())

    def get(self, tag):
        """Return the value for a field wih given tag.

        If there is more than one value, only the first is returned.  If the
        tag is not present, KeyError is raised.
        """
        for t, v in self.items:
            if t == tag:
                return v
        else:
            raise KeyError(tag)
    __getitem__ = get

    def get_all(self, tag):
        r = []
        for t, v in self.items:
            if t == tag:
                r.append(v)
        return r

    def as_dict(self):
        """Return a dict containing the unique values of the stanza.
        """
        d = {}
        for tag, value in self.items:
            d[tag] = value
        return d