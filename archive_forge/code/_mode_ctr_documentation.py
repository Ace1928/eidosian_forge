import struct
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.py3compat import _copy_bytes, is_native_int
from Cryptodome.Util.number import long_to_bytes
Nonce; not available if there is a fixed suffix