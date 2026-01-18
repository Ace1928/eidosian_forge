from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import io
import logging
import re
import sys
from cmakelang import lex
from cmakelang import markup
from cmakelang.common import UserError
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import PositionalGroupNode
from cmakelang.parse.common import FlowType, NodeType, TreeNode
from cmakelang.parse.util import comment_is_tag
from cmakelang.parse import simple_nodes
class AssertTypeDescriptor(object):

    def __init__(self, assert_type, hidden_name):
        self._assert_type = assert_type
        self._hidden_name = hidden_name

    def __get__(self, obj, objtype):
        return getattr(obj, self._hidden_name, None)

    def __set__(self, obj, value):
        assert isinstance(value, self._assert_type)
        setattr(obj, self._hidden_name, value)