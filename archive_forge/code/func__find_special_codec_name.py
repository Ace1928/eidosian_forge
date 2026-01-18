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
def _find_special_codec_name(self, encoding):
    try:
        requested_codec = codecs.getencoder(encoding)
    except LookupError:
        return None
    for name, codec in self._special_codecs:
        if codec == requested_codec:
            if '_' in name:
                name = ''.join([s.capitalize() for s in name.split('_')])
            return name
    return None