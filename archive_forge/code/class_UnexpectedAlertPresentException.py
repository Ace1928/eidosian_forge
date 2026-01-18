from typing import Optional
from typing import Sequence
class UnexpectedAlertPresentException(WebDriverException):
    """Thrown when an unexpected alert has appeared.

    Usually raised when  an unexpected modal is blocking the webdriver
    from executing commands.
    """

    def __init__(self, msg: Optional[str]=None, screen: Optional[str]=None, stacktrace: Optional[Sequence[str]]=None, alert_text: Optional[str]=None) -> None:
        super().__init__(msg, screen, stacktrace)
        self.alert_text = alert_text

    def __str__(self) -> str:
        return f'Alert Text: {self.alert_text}\n{super().__str__()}'