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

    __dataclass_fields__ contains a bunch of field objects recording how each field
    of the dataclass was initialized (mainly corresponding to the arguments passed to
    the "field" function). This node is used for the attributes of these field objects.

    If possible, coerces `arg` to a Python object.
    Otherwise, generates a sensible backup string.
    