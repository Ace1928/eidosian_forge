import itertools
from typing import Any, Callable, Dict, Set
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
class Resolver(object):
    """Resolver objects handle the process of looking up actual names and types.

  Unless noted otherwise, all resolve_* methods:
    * have a first namespace argument, mapping string to actual values
    * have a second types_namespace argument, mapping string to actual inferred
      types
    * specify names as QN objects
    * specify types as a Set of inferred types

  Unless noted otherwise, all resolve_* methods must return either:
    * a set of `type` objects
    * None
  """

    def res_name(self, ns, types_ns, name):
        """Resolves the type/value an external (e.g. closure, global) variable.

    Args:
      ns: namespace
      types_ns: types namespace
      name: symbol name
    Returns:
      Tuple (type, static_value). The first element is the type to use for
      inferrence. The second is the static value to use. Return None to treat it
      as unknown.
    """
        raise NotImplementedError('subclasses must implement')

    def res_value(self, ns, value):
        """Resolves the type a literal or static value."""
        raise NotImplementedError('subclasses must implement')

    def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        """Resolves the type of a (possibly annotated) function argument.

    Args:
      ns: namespace
      types_ns: types namespace
      f_name: str, the function name
      name: str, the argument name
      type_anno: the type annotating the argument, if any
      f_is_local: bool, whether the function is a local function
    Returns:
      Set of the argument types.
    """
        raise NotImplementedError('subclasses must implement')

    def res_call(self, ns, types_ns, node, f_type, args, keywords):
        """Resolves the return type an external function or method call.

    Args:
      ns: namespace
      types_ns: types namespace
      node: str, the function name
      f_type: types of the actual function being called, if known
      args: types of each respective argument in node.args
      keywords: types of each respective argument in node.keywords

    Returns:
      Tuple (return_type, side_effect_types). The first element is just the
      return types of the function. The second element is a map from
      argument names to sets of types, and allow modelling side effects of
      functions (for example via global or nonlocal).
    """
        raise NotImplementedError('subclasses must implement')

    def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
        """Resolves the return type of slice operation."""
        raise NotImplementedError('subclasses must implement')

    def res_compare(self, ns, types_ns, node, left, right):
        """Resolves the return type of a unary operation."""
        raise NotImplementedError('subclasses must implement')

    def res_unop(self, ns, types_ns, node, opnd):
        """Resolves the return type of a unary operation."""
        raise NotImplementedError('subclasses must implement')

    def res_binop(self, ns, types_ns, node, left, right):
        """Resolves the return type of a binary operation."""
        raise NotImplementedError('subclasses must implement')

    def res_list_literal(self, ns, elt_types):
        """Resolves the type of a list literal from its elements."""
        raise NotImplementedError('subclasses must implement')