import types
from pure_eval.utils import of_type, CannotEval
class _foo:
    __slots__ = ['foo']
    method = lambda: 0