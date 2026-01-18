from typing import Type
from lightning_utilities.core.enums import StrEnum
from typing_extensions import Literal
class ClassificationTaskNoMultilabel(EnumStr):
    """Enum to represent the different tasks in classification metrics.

    >>> "multilabel" in list(ClassificationTaskNoMultilabel)
    False

    """

    @staticmethod
    def _name() -> str:
        return 'Classification'
    BINARY = 'binary'
    MULTICLASS = 'multiclass'