import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
class _NodesTreeNode:
    _ChildrenGroup = namedtuple('_ChildrenGroup', 'prefix children line_offset last_line_offset_leaf')

    def __init__(self, tree_node, parent=None, indentation=0):
        self.tree_node = tree_node
        self._children_groups = []
        self.parent = parent
        self._node_children = []
        self.indentation = indentation

    def finish(self):
        children = []
        for prefix, children_part, line_offset, last_line_offset_leaf in self._children_groups:
            first_leaf = _get_next_leaf_if_indentation(children_part[0].get_first_leaf())
            first_leaf.prefix = prefix + first_leaf.prefix
            if line_offset != 0:
                try:
                    _update_positions(children_part, line_offset, last_line_offset_leaf)
                except _PositionUpdatingFinished:
                    pass
            children += children_part
        self.tree_node.children = children
        for node in children:
            node.parent = self.tree_node
        for node_child in self._node_children:
            node_child.finish()

    def add_child_node(self, child_node):
        self._node_children.append(child_node)

    def add_tree_nodes(self, prefix, children, line_offset=0, last_line_offset_leaf=None):
        if last_line_offset_leaf is None:
            last_line_offset_leaf = children[-1].get_last_leaf()
        group = self._ChildrenGroup(prefix, children, line_offset, last_line_offset_leaf)
        self._children_groups.append(group)

    def get_last_line(self, suffix):
        line = 0
        if self._children_groups:
            children_group = self._children_groups[-1]
            last_leaf = _get_previous_leaf_if_indentation(children_group.last_line_offset_leaf)
            line = last_leaf.end_pos[0] + children_group.line_offset
            if _ends_with_newline(last_leaf, suffix):
                line -= 1
        line += len(split_lines(suffix)) - 1
        if suffix and (not suffix.endswith('\n')) and (not suffix.endswith('\r')):
            line += 1
        if self._node_children:
            return max(line, self._node_children[-1].get_last_line(suffix))
        return line

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.tree_node)