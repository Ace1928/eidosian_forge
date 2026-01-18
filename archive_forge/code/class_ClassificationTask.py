from typing import Type
from lightning_utilities.core.enums import StrEnum
from typing_extensions import Literal
class ClassificationTask(EnumStr):
    """Enum to represent the different tasks in classification metrics.

    >>> "binary" in list(ClassificationTask)
    True

    """

    @staticmethod
    def _name() -> str:
        return 'Classification'
    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    MULTILABEL = 'multilabel'