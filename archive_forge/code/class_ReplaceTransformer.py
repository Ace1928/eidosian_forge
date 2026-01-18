import ast
import textwrap
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
class ReplaceTransformer(gast.NodeTransformer):
    """Replace AST nodes."""

    def __init__(self, replacements):
        """Create a new ReplaceTransformer.

    Args:
      replacements: A mapping from placeholder names to (lists of) AST nodes
          that these placeholders will be replaced by.
    """
        self.replacements = replacements
        self.in_replacements = False
        self.preserved_annos = {anno.Basic.DIRECTIVES, anno.Basic.EXTRA_LOOP_TEST, anno.Basic.ORIGIN, anno.Basic.SKIP_PROCESSING, anno.Static.ORIG_DEFINITIONS, 'function_context_name'}

    def _prepare_replacement(self, replaced, key):
        """Prepares a replacement AST that's safe to swap in for a node.

    Args:
      replaced: ast.AST, the node being replaced
      key: Hashable, the key of the replacement AST
    Returns:
      ast.AST, the replacement AST
    """
        repl = self.replacements[key]
        new_nodes = ast_util.copy_clean(repl, preserve_annos=self.preserved_annos)
        if isinstance(new_nodes, gast.AST):
            new_nodes = [new_nodes]
        return new_nodes

    def visit_Expr(self, node):
        new_value = self.visit(node.value)
        if new_value is node.value:
            return node
        return new_value

    def visit_keyword(self, node):
        if node.arg not in self.replacements:
            return self.generic_visit(node)
        repl = self._prepare_replacement(node, node.arg)
        if isinstance(repl, gast.keyword):
            return repl
        elif repl and isinstance(repl, (list, tuple)) and all((isinstance(r, gast.keyword) for r in repl)):
            return repl
        raise ValueError('a keyword argument may only be replaced by another keyword or a non-empty list of keywords. Found: {} for keyword {}'.format(repl, node.arg))

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name not in self.replacements:
            return node
        repl = self.replacements[node.name]
        if not isinstance(repl, (gast.Name, ast.Name)):
            raise ValueError('a function name can only be replaced by a Name node. Found: %s' % repl)
        node.name = repl.id
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if node.attr not in self.replacements:
            return node
        repl = self.replacements[node.attr]
        if not isinstance(repl, gast.Name):
            raise ValueError('An attribute can only be replaced by a Name node. Found: %s' % repl)
        node.attr = repl.id
        return node

    def visit_Name(self, node):
        if node.id not in self.replacements:
            return node
        new_nodes = self._prepare_replacement(node, node.id)
        if not new_nodes:
            return new_nodes
        adjuster = ContextAdjuster(type(node.ctx))
        for n in new_nodes:
            if hasattr(n, 'ctx'):
                adjuster.visit(n)
        if len(new_nodes) == 1:
            new_nodes, = new_nodes
        return new_nodes