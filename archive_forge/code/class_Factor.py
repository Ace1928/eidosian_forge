from __future__ import annotations
import logging # isort:skip
import typing as tp
from .bases import Init, SingleParameterizedProperty
from .container import Seq, Tuple
from .either import Either
from .primitive import String
from .singletons import Intrinsic
class Factor(SingleParameterizedProperty[FactorType]):
    """ Represents a single categorical factor. """

    def __init__(self, default: Init[FactorType]=Intrinsic, *, help: str | None=None) -> None:
        type_param = Either(L1Factor, L2Factor, L3Factor)
        super().__init__(type_param, default=default, help=help)