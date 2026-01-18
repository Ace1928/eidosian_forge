from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Util.py3compat import bchr
from . import TurboSHAKE128

        Produce more bytes of the digest.

        .. note::
            You cannot use :meth:`update` anymore after the first call to
            :meth:`read`.

        Args:
            length (integer): the amount of bytes this method must return

        :return: the next piece of XOF output (of the given length)
        :rtype: byte string
        