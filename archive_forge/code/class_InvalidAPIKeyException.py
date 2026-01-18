import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class InvalidAPIKeyException(AuthZeroException):
    """
    Invalid API Key Exception
    """
    base = 'Invalid API Key'
    concat_detail = True
    log_devel = True
    default_status_code = 403