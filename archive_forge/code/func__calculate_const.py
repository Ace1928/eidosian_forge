from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
def _calculate_const(self, node):
    if not self.reevaluate and node.constant_result is not ExprNodes.constant_value_not_set:
        return
    not_a_constant = ExprNodes.not_a_constant
    node.constant_result = not_a_constant
    children = self.visitchildren(node)
    for child_result in children.values():
        if type(child_result) is list:
            for child in child_result:
                if getattr(child, 'constant_result', not_a_constant) is not_a_constant:
                    return
        elif getattr(child_result, 'constant_result', not_a_constant) is not_a_constant:
            return
    try:
        node.calculate_constant_result()
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError):
        pass
    except Exception:
        import traceback, sys
        traceback.print_exc(file=sys.stdout)