from Cryptodome.Util.py3compat import bord
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
Return a copy ("clone") of the hash object.

        The copy will have the same internal state as the original hash
        object.
        This can be used to efficiently compute the digests of strings that
        share a common initial substring.

        :return: A hash object of the same type
        