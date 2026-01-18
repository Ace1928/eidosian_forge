from typing import Optional
from typing import Sequence
class ElementNotSelectableException(InvalidElementStateException):
    """Thrown when trying to select an unselectable element.

    For example, selecting a 'script' element.
    """