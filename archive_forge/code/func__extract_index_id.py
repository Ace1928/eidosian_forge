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
def _extract_index_id(self, index_node):
    base = index_node.base
    index = index_node.index
    if isinstance(index, ExprNodes.NameNode):
        index_val = index.name
    elif isinstance(index, ExprNodes.ConstNode):
        return None
    else:
        return None
    return (base.name, index_val)