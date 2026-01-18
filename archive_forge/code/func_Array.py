import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def Array(of_type, dim):
    return Collection(Traits([ArrayTrait, LenTrait, SliceTrait]), AnyType, AnyType, Array(of_type, dim - 1) if dim > 1 else of_type)