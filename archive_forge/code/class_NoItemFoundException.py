import traceback
from fastapi.exceptions import HTTPException
from lazyops.utils.logs import logger
from typing import Optional
class NoItemFoundException(ORMException):
    """
    No Item Found Exception
    """
    base = 'No Item Found'
    default_status_code = 404