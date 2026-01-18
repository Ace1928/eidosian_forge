from typing import Optional
from typing import Sequence
class NoSuchCookieException(WebDriverException):
    """No cookie matching the given path name was found amongst the associated
    cookies of the current browsing context's active document."""