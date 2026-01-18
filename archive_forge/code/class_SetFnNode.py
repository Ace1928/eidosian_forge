import collections
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
class SetFnNode(ArgGroupNode):

    def __init__(self):
        super(SetFnNode, self).__init__()
        self.varname = None
        self.value_group = None
        self.cache = None
        self.parent_scope = False