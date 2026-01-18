import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _update_parsed_node_tos(self, tree_node, keyword_token_indents):
    if tree_node.type == 'suite':
        def_leaf = tree_node.parent.children[0]
        new_tos = _NodesTreeNode(tree_node, indentation=keyword_token_indents[def_leaf.start_pos][-1])
        new_tos.add_tree_nodes('', list(tree_node.children))
        self._working_stack[-1].add_child_node(new_tos)
        self._working_stack.append(new_tos)
        self._update_parsed_node_tos(tree_node.children[-1], keyword_token_indents)
    elif _func_or_class_has_suite(tree_node):
        self._update_parsed_node_tos(tree_node.children[-1], keyword_token_indents)