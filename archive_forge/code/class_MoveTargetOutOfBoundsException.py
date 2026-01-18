from typing import Optional
from typing import Sequence
class MoveTargetOutOfBoundsException(WebDriverException):
    """Thrown when the target provided to the `ActionsChains` move() method is
    invalid, i.e. out of document."""