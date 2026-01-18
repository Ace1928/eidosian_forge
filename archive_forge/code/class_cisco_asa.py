from binascii import hexlify, unhexlify
from hashlib import md5
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import right_pad_string, to_unicode, repeat_string, to_bytes
from passlib.utils.binary import h64
from passlib.utils.compat import unicode, u, join_byte_values, \
import passlib.utils.handlers as uh
class cisco_asa(cisco_pix):
    """
    This class implements the password hash used by Cisco ASA/PIX 7.0 and newer (2005).
    Aside from a different internal algorithm, it's use and format is identical
    to the older :class:`cisco_pix` class.

    For passwords less than 13 characters, this should be identical to :class:`!cisco_pix`,
    but will generate a different hash for most larger inputs
    (See the `Format & Algorithm`_ section for the details).

    This class only allows passwords <= 32 bytes, anything larger
    will result in a :exc:`~passlib.exc.PasswordSizeError` if passed to :meth:`~cisco_asa.hash`,
    and be silently rejected if passed to :meth:`~cisco_asa.verify`.

    .. versionadded:: 1.7

    .. versionchanged:: 1.7.1

        Passwords > 32 bytes are now rejected / throw error instead of being silently truncated,
        to match Cisco behavior.  A number of :ref:`bugs <passlib-asa96-bug>` were fixed
        which caused prior releases to generate unverifiable hashes in certain cases.
    """
    name = 'cisco_asa'
    truncate_size = 32
    _is_asa = True