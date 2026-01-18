import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class InvalidRolesException(AuthZeroException):
    """
    Invalid Roles Exception
    """
    base = 'User does not have sufficient roles'
    concat_detail = False
    log_devel = True
    default_status_code = 403

    def __init__(self, roles: List[str], require_all: bool=False):
        """
        Initializes the invalid roles exception
        """
        detail = f'User does not have sufficient roles `{roles}`' if require_all else f'User does not have any of the roles `{roles}`'
        super().__init__(detail=detail)