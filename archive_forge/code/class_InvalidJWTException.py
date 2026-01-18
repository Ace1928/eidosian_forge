import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class InvalidJWTException(AuthZeroException):
    """
    Invalid JWT Exception
    """
    base = 'Invalid JWT'
    concat_detail = True
    log_devel = True
    default_status_code = 401