from __future__ import annotations
import logging # isort:skip
from abc import abstractmethod
from typing import Any
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
class CustomDimensional(Dimensional):
    """ A base class for units of measurement with an explicit basis.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
    basis = Required(Dict(String, Either(Tuple(Float, String), Tuple(Float, String, String))), help='\n    The basis defining the units of measurement.\n\n    This consists of a mapping between short unit names and their corresponding\n    scaling factors, TeX names and optional long names. For example, the basis\n    for defining angular units of measurement is:\n\n    .. code-block:: python\n\n        basis = {\n            "°":  (1,      "^\\circ",           "degree"),\n            "\'":  (1/60,   "^\\prime",          "minute"),\n            "\'\'": (1/3600, "^{\\prime\\prime}", "second"),\n        }\n    ')

    def is_known(self, unit: str) -> bool:
        return unit in self.basis