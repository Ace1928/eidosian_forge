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
def _pyucs4_to_number(self, node, py_type_name, func_arg):
    assert py_type_name in ('int', 'float')
    return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_int_from_UCS4' if py_type_name == 'int' else '__Pyx_double_from_UCS4', func_type=self.pyucs4_int_func_type if py_type_name == 'int' else self.pyucs4_double_func_type, args=[func_arg], py_name=py_type_name, is_temp=node.is_temp, result_is_used=node.result_is_used, utility_code=UtilityCode.load_cached('int_pyucs4' if py_type_name == 'int' else 'float_pyucs4', 'Builtins.c')).coerce_to(node.type, self.current_env())