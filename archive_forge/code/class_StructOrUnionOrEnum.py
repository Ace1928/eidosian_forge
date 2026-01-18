import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
class StructOrUnionOrEnum(BaseTypeByIdentity):
    _attrs_ = ('name',)
    forcename = None

    def build_c_name_with_marker(self):
        name = self.forcename or '%s %s' % (self.kind, self.name)
        self.c_name_with_marker = name + '&'

    def force_the_name(self, forcename):
        self.forcename = forcename
        self.build_c_name_with_marker()

    def get_official_name(self):
        assert self.c_name_with_marker.endswith('&')
        return self.c_name_with_marker[:-1]