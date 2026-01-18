import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def Str(rec=6):
    Next = Str(rec - 1) if rec else TypeVariable()
    return Collection(Traits([StrTrait, LenTrait, SliceTrait]), Integer(), Next, Next)