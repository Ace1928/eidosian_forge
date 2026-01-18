import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['variable_scope'])
class variable_scope:
    """A context manager for defining ops that creates variables (layers).

  @compatibility(TF2)
  Although it is a legacy `compat.v1` api,
  `tf.compat.v1.variable_scope` is mostly compatible with eager
  execution and `tf.function` as long as you combine it with the
  `tf.compat.v1.keras.utils.track_tf1_style_variables` decorator (though
  it will behave as if reuse is always set to `AUTO_REUSE`.)

  See the
  [model migration guide](
      https://www.tensorflow.org/guide/migrate/model_mapping)
  for more info on
  migrating code that relies on `variable_scope`-based variable reuse.

  When you use it with eager execution enabled but without
  `tf.compat.v1.keras.utils.track_tf1_style_variables`,
  `tf.compat.v1.variable_scope` will still be able to prefix the names
  of variables created within the scope but it will not enable variable reuse
  or error-raising checks around variable reuse (`get_variable` calls within
  it would always create new variables).

  Once you have switched away from `get_variable`-based variable reuse
  mechanisms, to switch to TF2 APIs you can just use
  `tf.name_scope` to prefix variable names.
  @end_compatibility

  This context manager validates that the (optional) `values` are from the same
  graph, ensures that graph is the default graph, and pushes a name scope and a
  variable scope.

  If `name_or_scope` is not None, it is used as is. If `name_or_scope` is None,
  then `default_name` is used.  In that case, if the same name has been
  previously used in the same scope, it will be made unique by appending `_N`
  to it.

  Variable scope allows you to create new variables and to share already created
  ones while providing checks to not create or share by accident. For details,
  see the [Variable Scope How To](https://tensorflow.org/guide/variables), here
  we present only a few basic examples.

  The Variable Scope works as expected when the Eager Execution is Disabled.

  ```python
  tf.compat.v1.disable_eager_execution()
  ```

  Simple example of how to create a new variable:

  ```python
  with tf.compat.v1.variable_scope("foo"):
      with tf.compat.v1.variable_scope("bar"):
          v = tf.compat.v1.get_variable("v", [1])
          assert v.name == "foo/bar/v:0"
  ```

  Simple example of how to reenter a premade variable scope safely:

  ```python
  with tf.compat.v1.variable_scope("foo") as vs:
    pass

  # Re-enter the variable scope.
  with tf.compat.v1.variable_scope(vs,
                         auxiliary_name_scope=False) as vs1:
    # Restore the original name_scope.
    with tf.name_scope(vs1.original_name_scope):
        v = tf.compat.v1.get_variable("v", [1])
        assert v.name == "foo/v:0"
        c = tf.constant([1], name="c")
        assert c.name == "foo/c:0"
  ```

  Keep in mind that the counters for `default_name` are discarded once the
  parent scope is exited. Therefore when the code re-enters the scope (for
  instance by saving it), all nested default_name counters will be restarted.

  For instance:

  ```python
  with tf.compat.v1.variable_scope("foo") as vs:
    with tf.compat.v1.variable_scope(None, default_name="bar"):
      v = tf.compat.v1.get_variable("a", [1])
      assert v.name == "foo/bar/a:0", v.name
    with tf.compat.v1.variable_scope(None, default_name="bar"):
      v = tf.compat.v1.get_variable("b", [1])
      assert v.name == "foo/bar_1/b:0"

  with tf.compat.v1.variable_scope(vs):
    with tf.compat.v1.variable_scope(None, default_name="bar"):
      v = tf.compat.v1.get_variable("c", [1])
      assert v.name == "foo/bar/c:0"   # Uses bar instead of bar_2!
  ```

  Basic example of sharing a variable AUTO_REUSE:

  ```python
  def foo():
    with tf.compat.v1.variable_scope("foo", reuse=tf.compat.v1.AUTO_REUSE):
      v = tf.compat.v1.get_variable("v", [1])
    return v

  v1 = foo()  # Creates v.
  v2 = foo()  # Gets the same, existing v.
  assert v1 == v2
  ```

  Basic example of sharing a variable with reuse=True:

  ```python
  with tf.compat.v1.variable_scope("foo"):
      v = tf.compat.v1.get_variable("v", [1])
  with tf.compat.v1.variable_scope("foo", reuse=True):
      v1 = tf.compat.v1.get_variable("v", [1])
  assert v1 == v
  ```

  Sharing a variable by capturing a scope and setting reuse:

  ```python
  with tf.compat.v1.variable_scope("foo") as scope:
      v = tf.compat.v1.get_variable("v", [1])
      scope.reuse_variables()
      v1 = tf.compat.v1.get_variable("v", [1])
  assert v1 == v
  ```

  To prevent accidental sharing of variables, we raise an exception when getting
  an existing variable in a non-reusing scope.

  ```python
  with tf.compat.v1.variable_scope("foo"):
      v = tf.compat.v1.get_variable("v", [1])
      v1 = tf.compat.v1.get_variable("v", [1])
      #  Raises ValueError("... v already exists ...").
  ```

  Similarly, we raise an exception when trying to get a variable that does not
  exist in reuse mode.

  ```python
  with tf.compat.v1.variable_scope("foo", reuse=True):
      v = tf.compat.v1.get_variable("v", [1])
      #  Raises ValueError("... v does not exists ...").
  ```

  Note that the `reuse` flag is inherited: if we open a reusing scope, then all
  its sub-scopes become reusing as well.

  A note about name scoping: Setting `reuse` does not impact the naming of other
  ops such as mult. See related discussion on
  [github#6189](https://github.com/tensorflow/tensorflow/issues/6189)

  Note that up to and including version 1.0, it was allowed (though explicitly
  discouraged) to pass False to the reuse argument, yielding undocumented
  behaviour slightly different from None. Starting at 1.1.0 passing None and
  False as reuse has exactly the same effect.

  A note about using variable scopes in multi-threaded environment: Variable
  scopes are thread local, so one thread will not see another thread's current
  scope. Also, when using `default_name`, unique scopes names are also generated
  only on a per thread basis. If the same name was used within a different
  thread, that doesn't prevent a new thread from creating the same scope.
  However, the underlying variable store is shared across threads (within the
  same graph). As such, if another thread tries to create a new variable with
  the same name as a variable created by a previous thread, it will fail unless
  reuse is True.

  Further, each thread starts with an empty variable scope. So if you wish to
  preserve name prefixes from a scope from the main thread, you should capture
  the main thread's scope and re-enter it in each thread. For e.g.

  ```
  main_thread_scope = variable_scope.get_variable_scope()

  # Thread's target function:
  def thread_target_fn(captured_scope):
    with variable_scope.variable_scope(captured_scope):
      # .... regular code for this thread


  thread = threading.Thread(target=thread_target_fn, args=(main_thread_scope,))
  ```
  """

    def __init__(self, name_or_scope, default_name=None, values=None, initializer=None, regularizer=None, caching_device=None, partitioner=None, custom_getter=None, reuse=None, dtype=None, use_resource=None, constraint=None, auxiliary_name_scope=True):
        """Initialize the context manager.

    Args:
      name_or_scope: `string` or `VariableScope`: the scope to open.
      default_name: The default name to use if the `name_or_scope` argument is
        `None`, this name will be uniquified. If name_or_scope is provided it
        won't be used and therefore it is not required and can be None.
      values: The list of `Tensor` arguments that are passed to the op function.
      initializer: default initializer for variables within this scope.
      regularizer: default regularizer for variables within this scope.
      caching_device: default caching device for variables within this scope.
      partitioner: default partitioner for variables within this scope.
      custom_getter: default custom getter for variables within this scope.
      reuse: `True`, None, or tf.compat.v1.AUTO_REUSE; if `True`, we go into
        reuse mode for this scope as well as all sub-scopes; if
        tf.compat.v1.AUTO_REUSE, we create variables if they do not exist, and
        return them otherwise; if None, we inherit the parent scope's reuse
        flag. When eager execution is enabled, new variables are always created
        unless an EagerVariableStore or template is currently active.
      dtype: type of variables created in this scope (defaults to the type in
        the passed scope, or inherited from parent scope).
      use_resource: If False, all variables will be regular Variables. If True,
        experimental ResourceVariables with well-defined semantics will be used
        instead. Defaults to False (will later change to True). When eager
        execution is enabled this argument is always forced to be True.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      auxiliary_name_scope: If `True`, we create an auxiliary name scope with
        the scope. If `False`, we don't create it. Note that the argument is not
        inherited, and it only takes effect for once when creating. You should
        only use it for re-entering a premade variable scope.

    Returns:
      A scope that can be captured and reused.

    Raises:
      ValueError: when trying to reuse within a create scope, or create within
        a reuse scope.
      TypeError: when the types of some arguments are not appropriate.
    """
        self._name_or_scope = name_or_scope
        self._default_name = default_name
        self._values = values
        self._initializer = initializer
        self._regularizer = regularizer
        self._caching_device = caching_device
        self._partitioner = partitioner
        self._custom_getter = custom_getter
        self._reuse = reuse
        self._dtype = dtype
        self._use_resource = use_resource
        self._constraint = constraint
        if self._default_name is None and self._name_or_scope is None:
            raise TypeError('If default_name is None then name_or_scope is required')
        if self._reuse is False:
            self._reuse = None
        if not (self._reuse is True or self._reuse is None or self._reuse is AUTO_REUSE):
            raise ValueError('The reuse parameter must be True or False or None.')
        if self._values is None:
            self._values = []
        self._in_graph_mode = not context.executing_eagerly()
        if self._in_graph_mode:
            self._graph = ops._get_graph_from_inputs(self._values)
        self._cached_pure_variable_scope = None
        self._current_name_scope = None
        if not isinstance(auxiliary_name_scope, bool):
            raise TypeError('The auxiliary_name_scope must be `True` or `False`, while get {}'.format(auxiliary_name_scope))
        self._auxiliary_name_scope = auxiliary_name_scope

    def __enter__(self):
        if ops.get_default_graph().building_function:
            self._building_function = True
        else:
            self._building_function = False
        if self._in_graph_mode and (not self._building_function):
            self._graph_context_manager = self._graph.as_default()
            self._graph_context_manager.__enter__()
        if self._cached_pure_variable_scope is not None:
            if self._current_name_scope is not None:
                self._current_name_scope.__enter__()
            return self._cached_pure_variable_scope.__enter__()
        try:
            return self._enter_scope_uncached()
        except:
            if self._in_graph_mode and (not self._building_function) and (self._graph_context_manager is not None):
                self._graph_context_manager.__exit__(*sys.exc_info())
            raise

    def _enter_scope_uncached(self):
        """Enters the context manager when there is no cached scope yet.

    Returns:
      The entered variable scope.

    Raises:
      TypeError: A wrong type is passed as `scope` at __init__().
      ValueError: `reuse` is incorrectly set at __init__().
    """
        if self._auxiliary_name_scope:
            current_name_scope = None
        else:
            name_scope = ops.get_name_scope()
            if name_scope:
                name_scope += '/'
                current_name_scope = ops.name_scope(name_scope, skip_on_eager=False)
            else:
                current_name_scope = ops.name_scope(name_scope, skip_on_eager=False)
        if self._name_or_scope is not None:
            if not isinstance(self._name_or_scope, (VariableScope, str)):
                raise TypeError('VariableScope: name_or_scope must be a string or VariableScope.')
            if isinstance(self._name_or_scope, str):
                name_scope = self._name_or_scope
            else:
                name_scope = self._name_or_scope.name.split('/')[-1]
            if name_scope or current_name_scope:
                current_name_scope = current_name_scope or ops.name_scope(name_scope, skip_on_eager=False)
                try:
                    current_name_scope_name = current_name_scope.__enter__()
                except:
                    current_name_scope.__exit__(*sys.exc_info())
                    raise
                self._current_name_scope = current_name_scope
                if isinstance(self._name_or_scope, str):
                    old_name_scope = current_name_scope_name
                else:
                    old_name_scope = self._name_or_scope.original_name_scope
                pure_variable_scope = _pure_variable_scope(self._name_or_scope, reuse=self._reuse, initializer=self._initializer, regularizer=self._regularizer, caching_device=self._caching_device, partitioner=self._partitioner, custom_getter=self._custom_getter, old_name_scope=old_name_scope, dtype=self._dtype, use_resource=self._use_resource, constraint=self._constraint)
                try:
                    entered_pure_variable_scope = pure_variable_scope.__enter__()
                except:
                    pure_variable_scope.__exit__(*sys.exc_info())
                    raise
                self._cached_pure_variable_scope = pure_variable_scope
                return entered_pure_variable_scope
            else:
                self._current_name_scope = None
                pure_variable_scope = _pure_variable_scope(self._name_or_scope, reuse=self._reuse, initializer=self._initializer, regularizer=self._regularizer, caching_device=self._caching_device, partitioner=self._partitioner, custom_getter=self._custom_getter, dtype=self._dtype, use_resource=self._use_resource, constraint=self._constraint)
                try:
                    entered_pure_variable_scope = pure_variable_scope.__enter__()
                except:
                    pure_variable_scope.__exit__(*sys.exc_info())
                    raise
                self._cached_pure_variable_scope = pure_variable_scope
                return entered_pure_variable_scope
        else:
            if self._reuse:
                raise ValueError('reuse=True cannot be used without a name_or_scope')
            current_name_scope = current_name_scope or ops.name_scope(self._default_name, skip_on_eager=False)
            try:
                current_name_scope_name = current_name_scope.__enter__()
            except:
                current_name_scope.__exit__(*sys.exc_info())
                raise
            self._current_name_scope = current_name_scope
            unique_default_name = _get_unique_variable_scope(self._default_name)
            pure_variable_scope = _pure_variable_scope(unique_default_name, initializer=self._initializer, regularizer=self._regularizer, caching_device=self._caching_device, partitioner=self._partitioner, custom_getter=self._custom_getter, old_name_scope=current_name_scope_name, dtype=self._dtype, use_resource=self._use_resource, constraint=self._constraint)
            try:
                entered_pure_variable_scope = pure_variable_scope.__enter__()
            except:
                pure_variable_scope.__exit__(*sys.exc_info())
                raise
            self._cached_pure_variable_scope = pure_variable_scope
            return entered_pure_variable_scope

    def __exit__(self, type_arg, value_arg, traceback_arg):
        try:
            self._cached_pure_variable_scope.__exit__(type_arg, value_arg, traceback_arg)
        finally:
            try:
                if self._current_name_scope:
                    self._current_name_scope.__exit__(type_arg, value_arg, traceback_arg)
            finally:
                if self._in_graph_mode and (not self._building_function):
                    self._graph_context_manager.__exit__(type_arg, value_arg, traceback_arg)