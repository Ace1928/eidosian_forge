import inspect
import threading
import types
import gast
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.utils import ag_logging as logging
def _wrap_into_factory(nodes, entity_name, inner_factory_name, outer_factory_name, closure_vars, factory_args, future_features):
    """Wraps an AST into the body of a factory with consistent lexical context.

  The AST is expected to define some symbol with a name given by `entity_name`.

  This mechanism ensures that the resulting transformed entity has lexical
  scoping identical to that of the source entity, while allowing extra
  parametrization.

  Two nested factories achieve the following:

   1. The inner factory dynamically creates the entity represented by `nodes`.
   2. The inner factory is parametrized by a custom set of arguments.
   3. The inner factory has a closure identical to that of the transformed
       entity.
   4. The inner factory has local variables named like `args`, which `nodes` may
       use as additional parameters.
   5. The inner factory returns the variables given by `entity_name`.
   6. The outer factory is niladic.
   7. The outer factory has no closure.
   8. The outer factory creates the necessary lexical scope for the inner
       factory, so that the loaded code has the given configuration for
       closure/globals.
   9. The outer factory returns the inner factory.

  Roughly speaking, the following code is generated:

      from __future__ import future_feature_1
      from __future__ import future_feature_2
      ...

      def outer_factory():
        closure_var_1 = None
        closure_var_2 = None
        ...

        def inner_factory(arg_1, arg_2, ...):
          <<nodes>>
          return entity

        return inner_factory

  The lexical scoping is created using dummy symbol declarations which create
  local variables in the body of the outer factory, so that the Python parser
  correctly marks them as free non-global variables upon load (that is, it
  creates cell slots for each symbol. These symbols are initialized with None,
  but their values are not expected to be used; instead, the caller is expected
  to replace them with the cells of the source entity. For more details, see:
  https://docs.python.org/3/reference/executionmodel.html#binding-of-names

  Args:
    nodes: Tuple[ast.AST], the source code to wrap.
    entity_name: Union[Text, ast.AST], the name of the principal entity that
      `nodes` define.
    inner_factory_name: Text, the name of the inner factory.
    outer_factory_name: Text, the name of the outer factory.
    closure_vars: Iterable[Text], names of the closure variables for the inner
      factory.
    factory_args: Iterable[Text], names of additional arguments for the
      inner factory. Useful to configure variables that the converted code can
      use. Typically, these are modules.
    future_features: Iterable[Text], names of future statements to associate the
      code with.

  Returns:
    ast.AST
  """
    dummy_closure_defs = []
    for var_name in closure_vars:
        template = '\n      var_name = None\n    '
        dummy_closure_defs.extend(templates.replace(template, var_name=var_name))
    if future_features:
        future_imports = gast.ImportFrom(module='__future__', names=[gast.alias(name=name, asname=None) for name in future_features], level=0)
    else:
        future_imports = []
    factory_args = [gast.Name(name, ctx=gast.Param(), annotation=None, type_comment=None) for name in factory_args]
    template = '\n    future_imports\n    def outer_factory_name():\n      dummy_closure_defs\n      def inner_factory_name(factory_args):\n        entity_defs\n        return entity_name\n      return inner_factory_name\n  '
    return templates.replace(template, dummy_closure_defs=dummy_closure_defs, entity_defs=nodes, entity_name=entity_name, factory_args=factory_args, future_imports=future_imports, inner_factory_name=inner_factory_name, outer_factory_name=outer_factory_name)