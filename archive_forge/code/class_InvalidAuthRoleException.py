import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class InvalidAuthRoleException(AuthZeroException):
    """
    Invalid Auth Role Exception
    """
    base = 'Invalid Auth Role'
    concat_detail = True
    log_devel = True
    default_status_code = 403