import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
class BzrFormat:
    """Base class for all formats of things living in metadirs.

    This class manages the format string that is stored in the 'format'
    or 'branch-format' file.

    All classes for (branch-, repository-, workingtree-) formats that
    live in meta directories and have their own 'format' file
    (i.e. different from .bzr/branch-format) derive from this class,
    as well as the relevant base class for their kind
    (BranchFormat, WorkingTreeFormat, RepositoryFormat).

    Each format is identified by a "format" or "branch-format" file with a
    single line containing the base format name and then an optional list of
    feature flags.

    Feature flags are supported as of bzr 2.5. Setting feature flags on formats
    will render them inaccessible to older versions of bzr.

    :ivar features: Dictionary mapping feature names to their necessity
    """
    _present_features: Set[str] = set()

    def __init__(self):
        self.features = {}

    @classmethod
    def register_feature(cls, name):
        """Register a feature as being present.

        :param name: Name of the feature
        """
        if b' ' in name:
            raise ValueError('spaces are not allowed in feature names')
        if name in cls._present_features:
            raise FeatureAlreadyRegistered(name)
        cls._present_features.add(name)

    @classmethod
    def unregister_feature(cls, name):
        """Unregister a feature."""
        cls._present_features.remove(name)

    def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
        for name, necessity in self.features.items():
            if name in self._present_features:
                continue
            if necessity == b'optional':
                mutter('ignoring optional missing feature %s', name)
                continue
            elif necessity == b'required':
                raise MissingFeature(name)
            else:
                mutter('treating unknown necessity as require for %s', name)
                raise MissingFeature(name)

    @classmethod
    def get_format_string(cls):
        """Return the ASCII format string that identifies this format."""
        raise NotImplementedError(cls.get_format_string)

    @classmethod
    def from_string(cls, text):
        format_string = cls.get_format_string()
        if not text.startswith(format_string):
            raise AssertionError('Invalid format header {!r} for {!r}'.format(text, cls))
        lines = text[len(format_string):].splitlines()
        ret = cls()
        for lineno, line in enumerate(lines):
            try:
                necessity, feature = line.split(b' ', 1)
            except ValueError:
                raise errors.ParseFormatError(format=cls, lineno=lineno + 2, line=line, text=text)
            ret.features[feature] = necessity
        return ret

    def as_string(self):
        """Return the string representation of this format.
        """
        lines = [self.get_format_string()]
        lines.extend([item[1] + b' ' + item[0] + b'\n' for item in sorted(self.features.items())])
        return b''.join(lines)

    @classmethod
    def _find_format(klass, registry, kind, format_string):
        try:
            first_line = format_string[:format_string.index(b'\n') + 1]
        except ValueError:
            first_line = format_string
        try:
            cls = registry.get(first_line)
        except KeyError:
            raise errors.UnknownFormatError(format=first_line, kind=kind)
        return cls.from_string(format_string)

    def network_name(self):
        """A simple byte string uniquely identifying this format for RPC calls.

        Metadir branch formats use their format string.
        """
        return self.as_string()

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.features == other.features

    def _update_feature_flags(self, updated_flags):
        """Update the feature flags in this format.

        :param updated_flags: Updated feature flags
        """
        for name, necessity in updated_flags.items():
            if necessity is None:
                try:
                    del self.features[name]
                except KeyError:
                    pass
            else:
                self.features[name] = necessity