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
class _PythonFnFactory(object):
    """Helper object that wraps a Python function factory."""

    def __init__(self, name, freevars, extra_locals):
        """Creates a new factory for a Python function.

    Args:
      name: The function name.
      freevars: The list of non-global free variables for the function.
      extra_locals: Dict[Text, Any], names and values for custom variables that
        are accessible to the generated code as local variables.
    """
        self._name = name
        self._freevars = freevars
        self._extra_locals = extra_locals
        self._unbound_factory = None
        self.module = None
        self.source_map = None

    def create(self, nodes, namer, inner_factory_name='inner_factory', outer_factory_name='outer_factory', future_features=()):
        """Initializes a function."""
        if self._unbound_factory is not None:
            raise ValueError('double initialization; create a new object instead')
        inner_factory_name = namer.new_symbol(inner_factory_name, ())
        outer_factory_name = namer.new_symbol(outer_factory_name, ())
        nodes = _wrap_into_factory(nodes, self._name, inner_factory_name, outer_factory_name, self._freevars, self._extra_locals.keys(), future_features)
        module, _, source_map = loader.load_ast(nodes, include_source_map=True)
        outer_factory = getattr(module, outer_factory_name)
        self._unbound_factory = outer_factory()
        self.module = module
        self.source_map = source_map

    def instantiate(self, globals_, closure, defaults=None, kwdefaults=None):
        """Creates a new function instance."""
        if self._unbound_factory is None:
            raise ValueError('call create first')
        factory_code = self._unbound_factory.__code__
        factory_freevars = factory_code.co_freevars
        closure_map = dict(zip(self._freevars, closure))
        factory_closure = tuple((closure_map[name] for name in factory_code.co_freevars))
        if len(factory_closure) != len(closure):
            raise ValueError('closure mismatch, requested {}, but source function had {}'.format(self._freevars, factory_freevars))
        bound_factory = types.FunctionType(code=factory_code, globals=globals_, name=self._name, argdefs=(), closure=factory_closure)
        new_fn = bound_factory(**self._extra_locals)
        if defaults:
            new_fn.__defaults__ = defaults
        if kwdefaults:
            new_fn.__kwdefaults__ = kwdefaults
        return new_fn