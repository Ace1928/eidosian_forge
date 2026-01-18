from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.common import InternalError
from cmakelang.parse.printer import dump_tree_tostr
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode

    Parser for the commands that take conditional arguments. Similar to the
    standard parser but it understands parentheses and can generate
    parenthentical groups::

        while(CONDITION1 AND (CONDITION2 OR CONDITION3)
              OR (CONDITION3 AND (CONDITION4 AND CONDITION5)
              OR CONDITION6)
    