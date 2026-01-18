from typing import Optional
from typing import Sequence
class NoAlertPresentException(WebDriverException):
    """Thrown when switching to no presented alert.

    This can be caused by calling an operation on the Alert() class when
    an alert is not yet on the screen.
    """