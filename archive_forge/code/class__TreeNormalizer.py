from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import re
from pasta.augment import errors
from pasta.base import formatting as fmt
class _TreeNormalizer(ast.NodeTransformer):
    """Replaces all op nodes with unique instances."""

    def visit(self, node):
        if isinstance(node, _AST_OP_NODES):
            return node.__class__()
        return super(_TreeNormalizer, self).visit(node)