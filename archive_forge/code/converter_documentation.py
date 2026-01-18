import enum
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.util.tf_export import tf_export
Returns the unique directive argument for a symbol.

    See lang/directives.py for details on directives.

    Example:
       # Given a directive in the code:
       ag.foo_directive(bar, baz=1)

       # One can write for an AST node Name(id='bar'):
       get_definition_directive(node, ag.foo_directive, 'baz')

    Args:
      node: ast.AST, the node representing the symbol for which the directive
        argument is needed.
      directive: Callable[..., Any], the directive to search.
      arg: str, the directive argument to return.
      default: Any

    Raises:
      ValueError: if conflicting annotations have been found
    