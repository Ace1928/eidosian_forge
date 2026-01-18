from typing import Optional
from typing import Sequence
class NoSuchDriverException(WebDriverException):
    """Raised when driver is not specified and cannot be located."""

    def __init__(self, msg: Optional[str]=None, screen: Optional[str]=None, stacktrace: Optional[Sequence[str]]=None) -> None:
        with_support = f'{msg}; {SUPPORT_MSG} {ERROR_URL}/driver_location'
        super().__init__(with_support, screen, stacktrace)