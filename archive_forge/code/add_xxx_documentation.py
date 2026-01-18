import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.util import (

  ::

    aux_source_directory(<dir> <variable>)

  :see: https://cmake.org/cmake/help/latest/command/aux_source_directory.html
  