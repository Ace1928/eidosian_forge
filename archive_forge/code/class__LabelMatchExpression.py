from typing import Dict, Union, Optional, TYPE_CHECKING
from ray.util.annotations import PublicAPI
class _LabelMatchExpression:
    """An expression used to select node by node's labels
    Attributes:
        key: the key of label
        operator: In、NotIn、Exists、DoesNotExist
    """

    def __init__(self, key: str, operator: Union[In, NotIn, Exists, DoesNotExist]):
        self.key = key
        self.operator = operator