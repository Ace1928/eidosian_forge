import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class InvalidAuthorizationException(AuthZeroException):
    """
    Invalid Authorization Exception
    """
    base = 'Invalid Authorization'
    concat_detail = True
    log_devel = True
    default_status_code = 403