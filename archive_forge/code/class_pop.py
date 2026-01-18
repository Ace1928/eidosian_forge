from typing import (
from torch import Tensor, nn
from ..microbatch import Batch
from .namespace import Namespace
from .tracker import current_skip_tracker
class pop:
    """The command to pop a skip tensor.

    ::

        def forward(self, input):
            skip = yield pop('name')
            return f(input) + skip

    Args:
        name (str): name of skip tensor

    Returns:
        the skip tensor previously stashed by another layer under the same name

    """
    __slots__ = ('name',)

    def __init__(self, name: str) -> None:
        self.name = name