import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class AuthZeroException(fastapi.HTTPException):
    """
    Base Auth Zero Exception
    """
    base: Optional[str] = None
    concat_detail: Optional[bool] = True
    verbose: Optional[int] = None
    default_status_code: Optional[int] = 400
    log_error: Optional[bool] = False
    log_devel: Optional[bool] = False

    def __init__(self, detail: str=None, error: Optional[Exception]=None, status_code: Optional[int]=None, **kwargs):
        """
        Constructor
        """
        from ..configs import settings
        message = ''
        if not detail or self.concat_detail:
            message += f'{self.base}: ' or ''
        if detail:
            message += detail
        if error and self.verbose is not None:
            if self.verbose >= 1:
                message += f'\nError: {error}'
            elif self.verbose <= 5:
                message += f'\nError: {error}\nTraceback: {traceback.format_exc()}'
        status_code = status_code or self.default_status_code
        super().__init__(status_code=status_code, detail=message, **kwargs)
        if self.log_error or (self.log_devel and settings.is_development_env):
            self.display(message, status_code)

    def display(self, message: str, status_code: int):
        """
        Displays the error
        """
        from ..utils.lazy import logger
        logger.error(f'[{self.__class__.__name__} - {status_code}]: {message}')