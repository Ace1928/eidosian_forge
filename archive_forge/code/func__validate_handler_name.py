import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedTypeError, PasslibWarning
from passlib.ifc import PasswordHash
from passlib.utils import (
from passlib.utils.compat import unicode_or_str
from passlib.utils.decor import memoize_single_value
def _validate_handler_name(name):
    """helper to validate handler name

    :raises ValueError:
        * if empty name
        * if name not lower case
        * if name contains double underscores
        * if name is reserved (e.g. ``context``, ``all``).
    """
    if not name:
        raise ValueError('handler name cannot be empty: %r' % (name,))
    if name.lower() != name:
        raise ValueError('name must be lower-case: %r' % (name,))
    if not _name_re.match(name):
        raise ValueError('invalid name (must be 3+ characters,  begin with a-z, and contain only underscore, a-z, 0-9): %r' % (name,))
    if '__' in name:
        raise ValueError('name may not contain double-underscores: %r' % (name,))
    if name in _forbidden_names:
        raise ValueError('that name is not allowed: %r' % (name,))
    return True