from typing import Optional, Tuple, ClassVar, Sequence
from .utils import Serialize
class NonTerminal(Symbol):
    __serialize_fields__ = ('name',)
    is_term: ClassVar[bool] = False