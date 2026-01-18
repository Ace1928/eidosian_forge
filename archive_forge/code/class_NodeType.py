from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import common
from cmakelang import lex
from cmakelang.parse.util import (
class NodeType(common.EnumObject):
    """
  Enumeration for AST nodes
  """
    _id_map = {}