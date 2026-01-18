from collections import OrderedDict
from textwrap import dedent
import operator
from . import ExprNodes
from . import Nodes
from . import PyrexTypes
from . import Builtin
from . import Naming
from .Errors import error, warning
from .Code import UtilityCode, TempitaUtilityCode, PyxCodeWriter
from .Visitor import VisitorTransform
from .StringEncoding import EncodedString
from .TreeFragment import TreeFragment
from .ParseTreeTransforms import NormalizeTree, SkipDeclarations
from .Options import copy_inherited_directives
class FieldRecordNode(ExprNodes.ExprNode):
    """
    __dataclass_fields__ contains a bunch of field objects recording how each field
    of the dataclass was initialized (mainly corresponding to the arguments passed to
    the "field" function). This node is used for the attributes of these field objects.

    If possible, coerces `arg` to a Python object.
    Otherwise, generates a sensible backup string.
    """
    subexprs = ['arg']

    def __init__(self, pos, arg):
        super(FieldRecordNode, self).__init__(pos, arg=arg)

    def analyse_types(self, env):
        self.arg.analyse_types(env)
        self.type = self.arg.type
        return self

    def coerce_to_pyobject(self, env):
        if self.arg.type.can_coerce_to_pyobject(env):
            return self.arg.coerce_to_pyobject(env)
        else:
            return self._make_string()

    def _make_string(self):
        from .AutoDocTransforms import AnnotationWriter
        writer = AnnotationWriter(description='Dataclass field')
        string = writer.write(self.arg)
        return ExprNodes.StringNode(self.pos, value=EncodedString(string))

    def generate_evaluation_code(self, code):
        return self.arg.generate_evaluation_code(code)