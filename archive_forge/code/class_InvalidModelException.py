import traceback
from fastapi.exceptions import HTTPException
from lazyops.utils.logs import logger
from typing import Optional
class InvalidModelException(ORMException):
    """
    Invalid Model Exception
    """
    base = 'Invalid Model'
    default_status_code = 400