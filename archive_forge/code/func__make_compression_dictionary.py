import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils
@classmethod
def _make_compression_dictionary(cls) -> bytes:
    """Returns a ``bytes`` object suitable for use as a dictionary for
        compression.
        """
    buf = []
    buf.append('numba')
    buf.append(cls.__class__.__name__)
    buf.extend(['True', 'False'])
    for k, opt in cls.options.items():
        buf.append(k)
        buf.append(str(opt.default))
    return ''.join(buf).encode()