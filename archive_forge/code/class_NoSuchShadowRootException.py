from typing import Optional
from typing import Sequence
class NoSuchShadowRootException(WebDriverException):
    """Thrown when trying to access the shadow root of an element when it does
    not have a shadow root attached."""