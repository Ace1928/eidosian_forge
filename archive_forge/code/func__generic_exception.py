import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def _generic_exception(err, name, message):
    if err in _error_to_exception:
        return _error_to_exception[err](name)
    else:
        return lzc_exc.ZFSGenericError(err, message, name)