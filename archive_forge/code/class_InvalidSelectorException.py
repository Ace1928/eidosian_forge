from typing import Optional
from typing import Sequence
class InvalidSelectorException(WebDriverException):
    """Thrown when the selector which is used to find an element does not
    return a WebElement.

    Currently this only happens when the selector is an xpath expression
    and it is either syntactically invalid (i.e. it is not a xpath
    expression) or the expression does not select WebElements (e.g.
    "count(//input)").
    """

    def __init__(self, msg: Optional[str]=None, screen: Optional[str]=None, stacktrace: Optional[Sequence[str]]=None) -> None:
        with_support = f'{msg}; {SUPPORT_MSG} {ERROR_URL}#invalid-selector-exception'
        super().__init__(with_support, screen, stacktrace)