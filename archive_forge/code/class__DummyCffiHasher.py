from __future__ import with_statement, absolute_import
import logging
import re
import types
from warnings import warn
from passlib import exc
from passlib.crypto.digest import MAX_UINT32
from passlib.utils import classproperty, to_bytes, render_bytes
from passlib.utils.binary import b64s_encode, b64s_decode
from passlib.utils.compat import u, unicode, bascii_to_str, uascii_to_str, PY2
import passlib.utils.handlers as uh
class _DummyCffiHasher:
    """
        dummy object to use as source of defaults when argon2_cffi isn't present.
        this tries to mimic the attributes of ``argon2.PasswordHasher()`` which the rest of
        this module reads.

        .. note:: values last synced w/ argon2 19.2 as of 2019-11-09
        """
    time_cost = 2
    memory_cost = 512
    parallelism = 2
    salt_len = 16
    hash_len = 16