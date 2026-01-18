from typing import Optional
from typing import Sequence
class ElementClickInterceptedException(WebDriverException):
    """The Element Click command could not be completed because the element
    receiving the events is obscuring the element that was requested to be
    clicked."""