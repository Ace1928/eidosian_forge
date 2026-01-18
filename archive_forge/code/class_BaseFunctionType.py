import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
class BaseFunctionType(BaseType):
    _attrs_ = ('args', 'result', 'ellipsis', 'abi')

    def __init__(self, args, result, ellipsis, abi=None):
        self.args = args
        self.result = result
        self.ellipsis = ellipsis
        self.abi = abi
        reprargs = [arg._get_c_name() for arg in self.args]
        if self.ellipsis:
            reprargs.append('...')
        reprargs = reprargs or ['void']
        replace_with = self._base_pattern % (', '.join(reprargs),)
        if abi is not None:
            replace_with = replace_with[:1] + abi + ' ' + replace_with[1:]
        self.c_name_with_marker = self.result.c_name_with_marker.replace('&', replace_with)