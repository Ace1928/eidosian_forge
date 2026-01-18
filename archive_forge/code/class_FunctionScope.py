from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.operators import variables
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import nest
class FunctionScope(object):
    """Context manager that wraps the body of a converted function.

  This context manager handles various operations related to the scope of a
  function:
    * optional TF name scopes - these name scopes match the name of the
        function, for easy visualization in tensorBoard;
    * optional automatic control dependencies - this adds the same mechanism
        for control dependencies that is used by `@tf.function`; it can be
        optionally enabled when using `tf.autograph.to_graph`;
    * tracking of autograph conversion state (whether it's enabled by the user,
        conversion options;
  """

    def __init__(self, function_name, scope_name, options):
        self.name = scope_name
        self.options = options
        if options.user_requested:
            self.autograph_ctx = ag_ctx.ControlStatusCtx(ag_ctx.Status.ENABLED, options)
        self.callopts = options.call_options()
        use_name_scope = options.uses(converter.Feature.NAME_SCOPES)
        self.use_name_scope = use_name_scope
        if use_name_scope:
            self.name_scope = ops.name_scope(self._sanitize(function_name))
        use_auto_deps = self.options.uses(converter.Feature.AUTO_CONTROL_DEPS)
        self.use_auto_deps = use_auto_deps
        if use_auto_deps:
            self.autodeps_scope = auto_control_deps.AutomaticControlDependencies()
            self._return_value_marked = False

    def _sanitize(self, name):
        """See https://www.tensorflow.org/api_docs/python/tf/Graph#name_scope."""
        if name and name.startswith('_'):
            name = 'fn' + name
        return name

    def __enter__(self):
        if self.options.user_requested:
            self.autograph_ctx.__enter__()
        if self.use_name_scope:
            self.name_scope.__enter__()
        if self.use_auto_deps:
            self.autodeps_scope.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.options.user_requested:
            self.autograph_ctx.__exit__(exc_type, exc_val, exc_tb)
        if self.use_name_scope:
            self.name_scope.__exit__(exc_type, exc_val, exc_tb)
        if self.use_auto_deps:
            self.autodeps_scope.__exit__(exc_type, exc_val, exc_tb)

    def ret(self, value, did_return):
        """Marks a value as returned from the function guarded by the scope."""
        del did_return
        if isinstance(value, variables.UndefinedReturnValue):
            return None
        if self.use_auto_deps:
            self._return_value_marked = True
            if value is None:
                return None

            def _mark_return_if_tensor(t):
                if tensor_util.is_tf_type(t):
                    return self.autodeps_scope.mark_as_return(t)
                return t
            value = nest.map_structure(_mark_return_if_tensor, value)
        return value