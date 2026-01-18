import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
class BaseTypeByIdentity(object):
    is_array_type = False
    is_raw_function = False

    def get_c_name(self, replace_with='', context='a C file', quals=0):
        result = self.c_name_with_marker
        assert result.count('&') == 1
        replace_with = replace_with.strip()
        if replace_with:
            if replace_with.startswith('*') and '&[' in result:
                replace_with = '(%s)' % replace_with
            elif not replace_with[0] in '[(':
                replace_with = ' ' + replace_with
        replace_with = qualify(quals, replace_with)
        result = result.replace('&', replace_with)
        if '$' in result:
            raise VerificationError("cannot generate '%s' in %s: unknown type name" % (self._get_c_name(), context))
        return result

    def _get_c_name(self):
        return self.c_name_with_marker.replace('&', '')

    def has_c_name(self):
        return '$' not in self._get_c_name()

    def is_integer_type(self):
        return False

    def get_cached_btype(self, ffi, finishlist, can_delay=False):
        try:
            BType = ffi._cached_btypes[self]
        except KeyError:
            BType = self.build_backend_type(ffi, finishlist)
            BType2 = ffi._cached_btypes.setdefault(self, BType)
            assert BType2 is BType
        return BType

    def __repr__(self):
        return '<%s>' % (self._get_c_name(),)

    def _get_items(self):
        return [(name, getattr(self, name)) for name in self._attrs_]