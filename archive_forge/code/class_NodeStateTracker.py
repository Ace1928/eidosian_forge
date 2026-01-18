import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
class NodeStateTracker(object):
    """Base class for general-purpose Python code transformation.

  This abstract class provides helpful functions, like state tracking within
  the scope of arbitrary node, helpers for processing code blocks, debugging,
  mapping of transformed code to original code, and others.

  Scope-local state tracking: to keep state across nodes, at the level of
  (possibly nested) scopes, use enter/exit_local_scope and set/get_local.
  You must call enter/exit_local_scope manually, but the transformer detects
  when they are not properly paired.

  The transformer allows keeping state across calls that is local
  to arbitrary nodes and their descendants, using the self.state attribute.
  Multiple independent scopes are allowed and automatically constructed.

  For example, to keep track of the `If` node that encloses any `Name` node,
  one can write:

  ```
    class FooType(object):

      def __init__(self):
        self.foo_property = None

    class DummyTransformer(NodeStateTracker, ast.NodeTransformer):

      def visit_If(self, node):
        self.state[FooType].enter()
        self.state[FooType].foo_property = node
        node = self.veneric_visit(node)
        self.state[FooType].exit()
        return node

      def visit_Name(self, node):
        self.state[FooType].foo_property  # will hold the innermost enclosing if
  ```

  Alternatively, the `enter()`/`exit()` calls can be managed by a `with`
  statement:

  ```
      def visit_If(self, node):
        with self.state[FooType] as foo:
          foo.foo_property = node
          return self.generic_visit(node)
  ```
  """

    def __init__(self, ctx):
        """Initialize the transformer.

    Subclasses should call this.

    Args:
      ctx: A Context object.
    """
        self._lineno = 0
        self._col_offset = 0
        self.ctx = ctx
        self.state = _State()

    def debug_print(self, node):
        """Helper method useful for debugging. Prints the AST."""
        if __debug__:
            print(pretty_printer.fmt(node))
        return node

    def debug_print_src(self, node):
        """Helper method useful for debugging. Prints the AST as code."""
        if __debug__:
            print(parser.unparse(node))
        return node

    def visit_block(self, nodes, before_visit=None, after_visit=None):
        """A more powerful version of generic_visit for statement blocks.

    An example of a block is the body of an if statement.

    This function allows specifying a postprocessing callback (the
    after_visit argument) argument which can be used to move nodes to a new
    destination. This is done by after_visit by returning a non-null
    second return value, e.g. return new_node, new_destination.

    For example, a transformer could perform the following move:

        foo()
        bar()
        baz()

        foo()
        if cond:
          bar()
          baz()

    The above could be done with a postprocessor of this kind:

        def after_visit(node):
          if node_is_function_call(bar):
            new_container_node = build_cond()
            new_container_node.body.append(node)
            return new_container_node, new_container_node.body
          else:
            # Once we set a new destination, all subsequent items will be
            # moved to it, so we don't need to explicitly handle baz.
            return node, None

    Args:
      nodes: enumerable of AST node objects. If None, the function returns None.
      before_visit: optional callable that is called before visiting each item
        in nodes
      after_visit: optional callable that takes in an AST node and returns a
        tuple (new_node, new_destination). It is called after visiting each item
        in nodes. Is used in the same was as the
          visit_* methods: new_node will replace the node; if not None,
            new_destination must be a list, and subsequent nodes will be placed
            in this list instead of the list returned by visit_block.

    Returns:
      A list of AST node objects containing the transformed items fron nodes,
      except those nodes that have been relocated using after_visit.
    """
        if nodes is None:
            return None
        results = []
        node_destination = results
        for node in nodes:
            if before_visit:
                before_visit()
            replacement = self.visit(node)
            if after_visit and replacement:
                replacement, new_destination = after_visit(replacement)
            else:
                new_destination = None
            if replacement:
                if isinstance(replacement, (list, tuple)):
                    node_destination.extend(replacement)
                else:
                    node_destination.append(replacement)
            if new_destination is not None:
                node_destination = new_destination
        return results