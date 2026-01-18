from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, _copy_bytes
from Cryptodome.Util._raw_api import is_buffer
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.Util import _cpu_features
def _get_ghash_portable():
    api = _ghash_api_template.replace('%imp%', 'portable')
    lib = load_pycryptodome_raw_lib('Cryptodome.Hash._ghash_portable', api)
    result = _build_impl(lib, 'portable')
    return result