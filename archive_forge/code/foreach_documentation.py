import logging
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker, NodeType
from cmakelang.parse.common import TreeNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (

  ``parse_foreach()`` has a couple of forms:

  * the usual form
  * range form
  * in (lists/items) form

  This function is just the dispatcher

  :see: https://cmake.org/cmake/help/latest/command/foreach.html
  