from typing import Optional
from typing import Sequence
class UnknownMethodException(WebDriverException):
    """The requested command matched a known URL but did not match any methods
    for that URL."""