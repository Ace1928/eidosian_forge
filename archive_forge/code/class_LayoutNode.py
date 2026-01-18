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
class LayoutNode(object):
    """
  An element in the format/layout tree. The structure of the layout tree
  mirrors that of the parse tree. We could store this info the nodes of the
  parse tree itself but it's a little cleaner to keep the functionality
  separate I think.
  """
    _position = AssertTypeDescriptor(Cursor, '__position')
    _size = AssertTypeDescriptor(Cursor, '__size')

    def __init__(self, pnode):
        self.pnode = pnode
        self._position = Cursor(0, 0)
        self._size = Cursor(0, 0)
        self._reflow_valid = False
        self.statement_terminal = False
        self._parent = None
        self._children = []
        self._passno = -1
        self._colextent = 0
        self._rowextent = 0
        self._stmt_depth = 0
        self._subtree_depth = 0
        self._locked = False
        self._layout_passes = [(0, False)]
        self._wrap = False
        assert isinstance(pnode, TreeNode)

    def _index_in_parent(self):
        for idx, child in enumerate(self._parent.children):
            if child is self:
                return idx
        return -1

    def next_sibling(self):
        if self._parent is None:
            return None
        next_idx = self._index_in_parent() + 1
        if next_idx >= len(self._parent.children):
            return None
        return self._parent.children[next_idx]

    @property
    def name(self):
        """
    The class name of the derived node type.
    """
        return self.__class__.__name__

    @property
    def passno(self):
        """
    The active pass-number which contributed the current layout of the
    subtree rooted at this node.
    """
        return self._passno

    @property
    def colextent(self):
        """
    The column index of the right-most character in the layout of the
    subtree rooted at this node. In other words, the width of the
    bounding box for the subtree rooted at this node.
    """
        return self._colextent

    @property
    def reflow_valid(self):
        """
    A boolean flag indicating whether or not the current layout is accepted.
    If False, then further layout passes are required.
    """
        return self._reflow_valid

    @property
    def position(self):
        """
    A cursor with the (row,col) of the first (i.e. top,left) character in the
    subtree rooted at this node.
    """
        return Cursor(*self._position)

    @property
    def node_type(self):
        """
    Return the `NodeType` of the corresponding parser node that generated
    this layout node.
    """
        return self.pnode.node_type

    @property
    def children(self):
        """
    A list of children layout nodes
    """
        return self._children

    @property
    def rowextent(self):
        return self._rowextent

    def __repr__(self):
        boolmap = {True: 'T', False: 'F'}
        return '{}({}),(passno={},wrap={},ok={}) pos:({},{}) ext:({},{})'.format(self.__class__.__name__, self.node_type.name, self._passno, boolmap[self._wrap], boolmap[self._reflow_valid], self.position[0], self.position[1], self.rowextent, self.colextent)

    def has_terminal_comment(self):
        """
    Return true if this node has a terminal line comment. In particular, this
    implies that no other node may be packed at the output cursor of this
    node's layout, and a line-wrap is required.
    """
        return False

    def get_depth(self):
        """
    Compute and return the depth of the subtree rooted at this node. The
    depth of the tree is the depth of the deepest (leaf) descendant.
    """
        if self._children:
            return 1 + max((child.get_depth() for child in self._children))
        return 1

    def lock(self, config, stmt_depth=0):
        """
    Lock the tree structure (topology) and prevent further updates. This is
    mostly for sanity checking. It also computes topological statistics such
    as `stmt_depth` and `subtree_depth`, and replaces the mutable list of
    children with an immuatable tuple.
    """
        self._stmt_depth = stmt_depth
        self._subtree_depth = self.get_depth()
        self._children = tuple(self._children)
        self._locked = True
        for child in self._children:
            child._parent = self
        if self.node_type == NodeType.STATEMENT:
            nextdepth = 1
        elif stmt_depth > 0:
            nextdepth = stmt_depth + 1
        else:
            nextdepth = 0
        for child in self._children:
            child.lock(config, nextdepth)

    def _reflow(self, stack_context, cursor, passno):
        """
    Overridden by concrete classes to implement the layout of characters.
    """
        raise NotImplementedError()

    def _validate_layout(self, stack_context, start_extent, end_extent):
        """
    Return true if the layout is acceptable according to several checks. For
    example, returns false if the content overflows the columnt limit.
    """
        config = stack_context.config
        if end_extent[1] > config.format.linewidth:
            return False
        size = end_extent - start_extent
        if not self._wrap:
            if size[0] > config.format.max_lines_hwrap:
                if not isinstance(self, (BodyNode, CommentNode, FlowControlNode)):
                    return False
            pathstr = get_pathstr(stack_context.node_path)
            if pathstr in config.format.always_wrap:
                return False
        return True

    def reflow(self, stack_context, cursor, parent_passno=0):
        """
    (re-)compute the layout of this node under the assumption that it should
    be placed at the given `cursor` on the current `parent_passno`.
    """
        assert self._locked
        assert isinstance(self.pnode, TreeNode)
        self._position = cursor.clone()
        outcursor = None
        layout_passes = stack_context.config.format.layout_passes.get(self.__class__.__name__, self._layout_passes)
        with stack_context.push_node(self):
            for passno, wrap in layout_passes:
                if passno > parent_passno:
                    break
                self._passno = passno
                self._wrap = wrap
                self._reflow_valid = True
                start_extent = cursor.clone()
                outcursor = self._reflow(stack_context, cursor.clone(), passno)
                end_extent = Cursor(outcursor[0], self._colextent)
                self._reflow_valid &= self._validate_layout(stack_context, start_extent, end_extent)
                if self._reflow_valid:
                    break
        assert outcursor is not None
        return outcursor

    def write(self, config, ctx):
        """
    Output text content given the currently configured layout.
    """
        for child in self._children:
            child.write(config, ctx)

    @staticmethod
    def create(pnode):
        """
    Create a new layout node associated with then given parser node.
    """
        if pnode.node_type in SCALAR_TYPES:
            return ScalarNode(pnode)
        if pnode.node_type == NodeType.ONOFFSWITCH:
            return OnOffSwitchNode(pnode)
        if pnode.node_type == NodeType.STATEMENT:
            return StatementNode(pnode)
        if pnode.node_type == NodeType.ATWORDSTATEMENT:
            return AtWordStatementNode(pnode)
        if pnode.node_type == NodeType.KWARGGROUP:
            return KwargGroupNode(pnode)
        if pnode.node_type == NodeType.ARGGROUP:
            return ArgGroupNode(pnode)
        if pnode.node_type in (NodeType.PARGGROUP, NodeType.FLAGGROUP):
            return PargGroupNode(pnode)
        if pnode.node_type == NodeType.PARENGROUP:
            return ParenGroupNode(pnode)
        if pnode.node_type == NodeType.BODY:
            return BodyNode(pnode)
        if pnode.node_type == NodeType.FLOW_CONTROL:
            return FlowControlNode(pnode)
        if pnode.node_type == NodeType.COMMENT:
            return CommentNode(pnode)
        if pnode.node_type == NodeType.WHITESPACE:
            return WhitespaceNode(pnode)
        if pnode.node_type in PAREN_TYPES:
            return ParenNode(pnode)
        raise RuntimeError('Unexpected node type')