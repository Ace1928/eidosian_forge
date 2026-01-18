from __future__ import print_function
import binascii
from Cryptodome.Util.py3compat import bord, bchr
Transform a string into a corresponding key.

    Example::

        >>> from Cryptodome.Util.RFC1751 import english_to_key
        >>> english_to_key('RAM LOIS GOAD CREW CARE HIT')
        b'66666666'

    Args:
      s (string): the string with the words separated by whitespace;
                  the number of words must be a multiple of 6.
    Return:
      A byte string.
    