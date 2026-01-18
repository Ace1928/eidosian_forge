from . import (
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform

    Everything related to defining an input/output argument for a ufunc

    type  - PyrexType
    type_constant  - str such as "NPY_INT8" representing numpy dtype constants
    