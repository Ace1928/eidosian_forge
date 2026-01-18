import base64
import struct
from os_ken.lib import addrconv
@classmethod
def _rev_lookup_type(cls, targ_cls):
    if cls._REV_TYPES is None:
        rev = dict(((v, k) for k, v in cls._TYPES.items()))
        cls._REV_TYPES = rev
    return cls._REV_TYPES[targ_cls]