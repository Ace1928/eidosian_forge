import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class ExpiredJWTException(AuthZeroException):
    """
    Expired JWT Exception
    """
    base = 'Expired JWT'
    concat_detail = True
    log_devel = True
    default_status_code = 401