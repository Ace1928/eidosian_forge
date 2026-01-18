import traceback
from fastapi.exceptions import HTTPException
from lazyops.utils.logs import logger
from typing import Optional
class MissingItemsException(ORMException):
    """
    Missing Items Exception
    """
    base = 'Not all Items were found for'
    default_status_code = 404