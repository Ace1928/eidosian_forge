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
def _handle_simple_function_all(self, node, pos_args):
    """Transform

        _result = all(p(x) for L in LL for x in L)

        into

        for L in LL:
            for x in L:
                if not p(x):
                    return False
        else:
            return True
        """
    return self._transform_any_all(node, pos_args, False)