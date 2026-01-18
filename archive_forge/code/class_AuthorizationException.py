import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class AuthorizationException(AuthZeroException):
    """
    Authorization Exception
    """
    base = 'Authorization Failed'
    concat_detail = True
    log_devel = True
    default_status_code = 403