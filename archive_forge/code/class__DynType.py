from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
from ._compatibility import compatibility
class _DynType:
    """
    _DynType defines a type which stands for the absence of type information.
    """

    def __init__(self):
        self.__name__ = '_DynType'

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __str__(self):
        return 'Dyn'

    def __repr__(self):
        return 'Dyn'