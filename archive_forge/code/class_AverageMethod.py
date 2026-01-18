from typing import Type
from lightning_utilities.core.enums import StrEnum
from typing_extensions import Literal
class AverageMethod(EnumStr):
    """Enum to represent average method.

    >>> None in list(AverageMethod)
    True
    >>> AverageMethod.NONE == None
    True
    >>> AverageMethod.NONE == 'none'
    True

    """

    @staticmethod
    def _name() -> str:
        return 'Average method'
    MICRO = 'micro'
    MACRO = 'macro'
    WEIGHTED = 'weighted'
    NONE = None
    SAMPLES = 'samples'