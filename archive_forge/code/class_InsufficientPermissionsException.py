import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class InsufficientPermissionsException(AuthZeroException):
    """
    Insufficient Permissions Exception
    """
    base = 'Insufficient Permissions'
    concat_detail = True
    log_devel = True
    default_status_code = 403