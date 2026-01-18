from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
class SliceIter(object):

    def __init__(self, slice_type, slice_result, ndim, code):
        self.slice_type = slice_type
        self.slice_result = slice_result
        self.code = code
        self.ndim = ndim