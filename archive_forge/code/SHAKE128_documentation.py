from Cryptodome.Util.py3compat import bord
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Hash.keccak import _raw_keccak_lib

        Compute the next piece of XOF output.

        .. note::
            You cannot use :meth:`update` anymore after the first call to
            :meth:`read`.

        Args:
            length (integer): the amount of bytes this method must return

        :return: the next piece of XOF output (of the given length)
        :rtype: byte string
        