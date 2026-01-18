import ast
import textwrap
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
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