import collections
import functools
import re
import threading
import warnings
import numpy as np
import wrapt
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import pywrap_tf_session as tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor
from tensorflow.python.ops import session_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class BaseSession(SessionInterface):
    """A class for interacting with a TensorFlow computation.

  The BaseSession enables incremental graph building with inline
  execution of Operations and evaluation of Tensors.
  """

    def __init__(self, target='', graph=None, config=None):
        """Constructs a new TensorFlow session.

    Args:
      target: (Optional) The TensorFlow execution engine to connect to.
      graph: (Optional) The graph to be used. If this argument is None, the
        default graph will be used.
      config: (Optional) ConfigProto proto used to configure the session. If no
        config is specified, the global default will be used. The global default
        can be configured via the tf.config APIs.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        creating the TensorFlow session.
      TypeError: If one of the arguments has the wrong type.
    """
        _python_session_create_counter.get_cell().increase_by(1)
        if graph is None:
            self._graph = ops.get_default_graph()
        else:
            if not isinstance(graph, ops.Graph):
                raise TypeError(f'Argument `graph` must be a tf.Graph, but got "{type(graph).__name__}"')
            self._graph = graph
        self._closed = False
        if target is not None:
            try:
                self._target = compat.as_bytes(target)
            except TypeError:
                if isinstance(target, config_pb2.ConfigProto):
                    raise TypeError(f'Argument `target` must be a string, but got "{type(target).__name__}". Did you do "Session(config)" instead of "Session(config=config)"?')
                raise TypeError(f'Argument `target` must be a string, but got "{type(target).__name__}"')
        else:
            self._target = None
        self._delete_lock = threading.Lock()
        self._dead_handles = []
        if config is None:
            config = context.context().config
        if not isinstance(config, config_pb2.ConfigProto):
            raise TypeError(f'Argument `config` must be a tf.ConfigProto, but got "{type(config).__name__}"')
        if mixed_precision_global_state.is_mixed_precision_graph_rewrite_enabled() and config.graph_options.rewrite_options.auto_mixed_precision != rewriter_config_pb2.RewriterConfig.OFF:
            new_config = config_pb2.ConfigProto()
            new_config.CopyFrom(config)
            new_config.graph_options.rewrite_options.auto_mixed_precision = rewriter_config_pb2.RewriterConfig.ON
            config = new_config
        elif config.graph_options.rewrite_options.auto_mixed_precision != rewriter_config_pb2.RewriterConfig.ON:
            mixed_precision_global_state.set_non_mixed_precision_session_created(True)
        self._config = config
        self._add_shapes = config.graph_options.infer_shapes
        self._session = None
        opts = tf_session.TF_NewSessionOptions(target=self._target, config=config)
        try:
            with self._graph._c_graph.get() as c_graph:
                self._session = tf_session.TF_NewSessionRef(c_graph, opts)
        finally:
            tf_session.TF_DeleteSessionOptions(opts)

    def list_devices(self):
        """Lists available devices in this session.

    ```python
    devices = sess.list_devices()
    for d in devices:
      print(d.name)
    ```

    Where:
      Each element in the list has the following properties
      name: A string with the full name of the device. ex:
          `/job:worker/replica:0/task:3/device:CPU:0`
      device_type: The type of the device (e.g. `CPU`, `GPU`, `TPU`.)
      memory_limit: The maximum amount of memory available on the device.
          Note: depending on the device, it is possible the usable memory could
          be substantially less.

    Raises:
      tf.errors.OpError: If it encounters an error (e.g. session is in an
      invalid state, or network errors occur).

    Returns:
      A list of devices in the session.
    """
        raw_device_list = tf_session.TF_SessionListDevices(self._session)
        device_list = []
        size = tf_session.TF_DeviceListCount(raw_device_list)
        for i in range(size):
            name = tf_session.TF_DeviceListName(raw_device_list, i)
            device_type = tf_session.TF_DeviceListType(raw_device_list, i)
            memory = tf_session.TF_DeviceListMemoryBytes(raw_device_list, i)
            incarnation = tf_session.TF_DeviceListIncarnation(raw_device_list, i)
            device_list.append(_DeviceAttributes(name, device_type, memory, incarnation))
        tf_session.TF_DeleteDeviceList(raw_device_list)
        return device_list

    def close(self):
        """Closes this session.

    Calling this method frees all resources associated with the session.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        closing the TensorFlow session.
    """
        if self._session and (not self._closed):
            self._closed = True
            tf_session.TF_CloseSession(self._session)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
        if self._session is not None:
            try:
                tf_session.TF_DeleteSession(self._session)
            except (AttributeError, TypeError):
                pass
            self._session = None

    @property
    def graph(self):
        """The graph that was launched in this session."""
        return self._graph

    @property
    def graph_def(self):
        """A serializable version of the underlying TensorFlow graph.

    Returns:
      A graph_pb2.GraphDef proto containing nodes for all of the Operations in
      the underlying TensorFlow graph.
    """
        return self._graph.as_graph_def(add_shapes=self._add_shapes)

    @property
    def sess_str(self):
        return self._target

    def as_default(self):
        """Returns a context manager that makes this object the default session.

    Use with the `with` keyword to specify that calls to
    `tf.Operation.run` or `tf.Tensor.eval` should be executed in
    this session.

    ```python
    c = tf.constant(..)
    sess = tf.compat.v1.Session()

    with sess.as_default():
      assert tf.compat.v1.get_default_session() is sess
      print(c.eval())
    ```

    To get the current default session, use `tf.compat.v1.get_default_session`.

    *N.B.* The `as_default` context manager *does not* close the
    session when you exit the context, and you must close the session
    explicitly.

    ```python
    c = tf.constant(...)
    sess = tf.compat.v1.Session()
    with sess.as_default():
      print(c.eval())
    # ...
    with sess.as_default():
      print(c.eval())

    sess.close()
    ```

    Alternatively, you can use `with tf.compat.v1.Session():` to create a
    session that is automatically closed on exiting the context,
    including when an uncaught exception is raised.

    *N.B.* The default session is a property of the current thread. If you
    create a new thread, and wish to use the default session in that
    thread, you must explicitly add a `with sess.as_default():` in that
    thread's function.

    *N.B.* Entering a `with sess.as_default():` block does not affect
    the current default graph. If you are using multiple graphs, and
    `sess.graph` is different from the value of
    `tf.compat.v1.get_default_graph`, you must explicitly enter a
    `with sess.graph.as_default():` block to make `sess.graph` the default
    graph.

    Returns:
      A context manager using this session as the default session.
    """
        return stack.default_session(self)

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        """Runs operations and evaluates tensors in `fetches`.

    This method runs one "step" of TensorFlow computation, by
    running the necessary graph fragment to execute every `Operation`
    and evaluate every `Tensor` in `fetches`, substituting the values in
    `feed_dict` for the corresponding input values.

    The `fetches` argument may be a single graph element, or an arbitrarily
    nested list, tuple, namedtuple, dict, or OrderedDict containing graph
    elements at its leaves.  A graph element can be one of the following types:

    * A `tf.Operation`.
      The corresponding fetched value will be `None`.
    * A `tf.Tensor`.
      The corresponding fetched value will be a numpy ndarray containing the
      value of that tensor.
    * A `tf.sparse.SparseTensor`.
      The corresponding fetched value will be a
      `tf.compat.v1.SparseTensorValue`
      containing the value of that sparse tensor.
    * A `get_tensor_handle` op.  The corresponding fetched value will be a
      numpy ndarray containing the handle of that tensor.
    * A `string` which is the name of a tensor or operation in the graph.

    The value returned by `run()` has the same shape as the `fetches` argument,
    where the leaves are replaced by the corresponding values returned by
    TensorFlow.

    Example:

    ```python
       a = tf.constant([10, 20])
       b = tf.constant([1.0, 2.0])
       # 'fetches' can be a singleton
       v = session.run(a)
       # v is the numpy array [10, 20]
       # 'fetches' can be a list.
       v = session.run([a, b])
       # v is a Python list with 2 numpy arrays: the 1-D array [10, 20] and the
       # 1-D array [1.0, 2.0]
       # 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
       MyData = collections.namedtuple('MyData', ['a', 'b'])
       v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
       # v is a dict with
       # v['k1'] is a MyData namedtuple with 'a' (the numpy array [10, 20]) and
       # 'b' (the numpy array [1.0, 2.0])
       # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
       # [10, 20].
    ```

    The optional `feed_dict` argument allows the caller to override
    the value of tensors in the graph. Each key in `feed_dict` can be
    one of the following types:

    * If the key is a `tf.Tensor`, the
      value may be a Python scalar, string, list, or numpy ndarray
      that can be converted to the same `dtype` as that
      tensor. Additionally, if the key is a
      `tf.compat.v1.placeholder`, the shape of
      the value will be checked for compatibility with the placeholder.
    * If the key is a
      `tf.sparse.SparseTensor`,
      the value should be a
      `tf.compat.v1.SparseTensorValue`.
    * If the key is a nested tuple of `Tensor`s or `SparseTensor`s, the value
      should be a nested tuple with the same structure that maps to their
      corresponding values as above.

    Each value in `feed_dict` must be convertible to a numpy array of the dtype
    of the corresponding key.

    The optional `options` argument expects a [`RunOptions`] proto. The options
    allow controlling the behavior of this particular step (e.g. turning tracing
    on).

    The optional `run_metadata` argument expects a [`RunMetadata`] proto. When
    appropriate, the non-Tensor output of this step will be collected there. For
    example, when users turn on tracing in `options`, the profiled info will be
    collected into this argument and passed back.

    Args:
      fetches: A single graph element, a list of graph elements, or a dictionary
        whose values are graph elements or lists of graph elements (described
        above).
      feed_dict: A dictionary that maps graph elements to values (described
        above).
      options: A [`RunOptions`] protocol buffer
      run_metadata: A [`RunMetadata`] protocol buffer

    Returns:
      Either a single value if `fetches` is a single graph element, or
      a list of values if `fetches` is a list, or a dictionary with the
      same keys as `fetches` if that is a dictionary (described above).
      Order in which `fetches` operations are evaluated inside the call
      is undefined.

    Raises:
      RuntimeError: If this `Session` is in an invalid state (e.g. has been
        closed).
      TypeError: If `fetches` or `feed_dict` keys are of an inappropriate type.
      ValueError: If `fetches` or `feed_dict` keys are invalid or refer to a
        `Tensor` that doesn't exist.
    """
        options_ptr = tf_session.TF_NewBufferFromString(compat.as_bytes(options.SerializeToString())) if options else None
        run_metadata_ptr = tf_session.TF_NewBuffer() if run_metadata else None
        try:
            result = self._run(None, fetches, feed_dict, options_ptr, run_metadata_ptr)
            if run_metadata:
                proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
                run_metadata.ParseFromString(compat.as_bytes(proto_data))
        finally:
            if run_metadata_ptr:
                tf_session.TF_DeleteBuffer(run_metadata_ptr)
            if options:
                tf_session.TF_DeleteBuffer(options_ptr)
        return result

    @deprecation.deprecated('2023-06-01', 'This function is deprecated and we do not expect adding newfunctionality to it. Please do not have your code dependingon this function.')
    def partial_run(self, handle, fetches, feed_dict=None):
        """Continues the execution with more feeds and fetches.

    NOTE: This function is deprecated and we do not expect adding new
    functionality to it. Please do not have your code depending on this
    function.

    This is EXPERIMENTAL and subject to change.

    To use partial execution, a user first calls `partial_run_setup()` and
    then a sequence of `partial_run()`. `partial_run_setup` specifies the
    list of feeds and fetches that will be used in the subsequent
    `partial_run` calls.

    The optional `feed_dict` argument allows the caller to override
    the value of tensors in the graph. See run() for more information.

    Below is a simple example:

    ```python
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    res = sess.partial_run(h, r2, feed_dict={c: res})
    ```

    Args:
      handle: A handle for a sequence of partial runs.
      fetches: A single graph element, a list of graph elements, or a dictionary
        whose values are graph elements or lists of graph elements (see
        documentation for `run`).
      feed_dict: A dictionary that maps graph elements to values (described
        above).

    Returns:
      Either a single value if `fetches` is a single graph element, or
      a list of values if `fetches` is a list, or a dictionary with the
      same keys as `fetches` if that is a dictionary
      (see documentation for `run`).

    Raises:
      tf.errors.OpError: Or one of its subclasses on error.
    """
        return self._run(handle, fetches, feed_dict, None, None)

    @deprecation.deprecated('2023-06-01', 'This function is deprecated and we do not expect adding newfunctionality to it. Please do not have your code dependingon this function.')
    def partial_run_setup(self, fetches, feeds=None):
        """Sets up a graph with feeds and fetches for partial run.

    NOTE: This function is deprecated and we do not expect adding new
    functionality to it. Please do not have your code depending on this
    function.

    This is EXPERIMENTAL and subject to change.

    Note that contrary to `run`, `feeds` only specifies the graph elements.
    The tensors will be supplied by the subsequent `partial_run` calls.

    Args:
      fetches: A single graph element, or a list of graph elements.
      feeds: A single graph element, or a list of graph elements.

    Returns:
      A handle for partial run.

    Raises:
      RuntimeError: If this `Session` is in an invalid state (e.g. has been
        closed).
      TypeError: If `fetches` or `feed_dict` keys are of an inappropriate type.
      tf.errors.OpError: Or one of its subclasses if a TensorFlow error happens.
    """

        def _feed_fn(feed):
            for tensor_type, _, _, feed_fn in _REGISTERED_EXPANSIONS:
                if isinstance(feed, tensor_type):
                    return feed_fn(feed)
            raise TypeError(f'Feed argument {feed} has invalid type "{type(feed).__name__}"')
        if self._closed:
            raise RuntimeError('Attempted to use a closed Session.')
        if self.graph.version == 0:
            raise RuntimeError('The Session graph is empty. Add operations to the graph before calling run().')
        if feeds is None:
            feeds = []
        feed_list = []
        is_list_feed = isinstance(feeds, (list, tuple))
        if not is_list_feed:
            feeds = [feeds]
        for feed in feeds:
            for subfeed in _feed_fn(feed):
                try:
                    subfeed_t = self.graph.as_graph_element(subfeed, allow_tensor=True, allow_operation=False)
                    feed_list.append(subfeed_t._as_tf_output())
                except Exception as e:
                    e.message = f'Cannot interpret argument `feed` key as Tensor: {e.message}'
                    e.args = (e.message,)
                    raise e
        fetch_handler = _FetchHandler(self._graph, fetches, {})

        def _setup_fn(session, feed_list, fetch_list, target_list):
            self._extend_graph()
            return tf_session.TF_SessionPRunSetup_wrapper(session, feed_list, fetch_list, target_list)
        final_fetches = [t._as_tf_output() for t in fetch_handler.fetches()]
        final_targets = [op._c_op for op in fetch_handler.targets()]
        return self._do_call(_setup_fn, self._session, feed_list, final_fetches, final_targets)

    def _run(self, handle, fetches, feed_dict, options, run_metadata):
        """Perform either run or partial_run, depending the presence of `handle`."""

        def _feed_fn(feed, feed_val):
            for tensor_type, _, feed_fn, _ in _REGISTERED_EXPANSIONS:
                if isinstance(feed, tensor_type):
                    return feed_fn(feed, feed_val)
            raise TypeError(f'{feed} in argument `feed_dict` has invalid type "{type(feed).__name__}"')
        if self._closed:
            raise RuntimeError('Attempted to use a closed Session.')
        if self.graph.version == 0:
            raise RuntimeError('The Session graph is empty. Add operations to the graph before calling run().')
        feed_dict_tensor = {}
        feed_map = {}
        feed_handles = {}
        if feed_dict:
            feed_dict = nest.flatten_dict_items(feed_dict)
            for feed, feed_val in feed_dict.items():
                for subfeed, subfeed_val in _feed_fn(feed, feed_val):
                    try:
                        subfeed_t = self.graph.as_graph_element(subfeed, allow_tensor=True, allow_operation=False)
                    except Exception as e:
                        raise TypeError(f'Cannot interpret feed_dict key as Tensor: {e.args[0]}')
                    if isinstance(subfeed_val, tensor.Tensor):
                        raise TypeError(f'The value of a feed cannot be a tf.Tensor object. Acceptable feed values include Python scalars, strings, lists, numpy ndarrays, or TensorHandles. For reference, the tensor object was {str(feed_val)} which was passed to the argument `feed_dict` with key {str(feed)}.')
                    subfeed_dtype = subfeed_t.dtype.as_numpy_dtype
                    if isinstance(subfeed_val, int) and _convert_to_numpy_obj(subfeed_dtype, subfeed_val) != subfeed_val:
                        raise TypeError(f'Type of feed value {str(subfeed_val)} with type ' + f'{str(type(subfeed_val))} is not compatible with Tensor type {str(subfeed_dtype)}. Try explicitly setting the type of the feed tensor to a larger type (e.g. int64).')
                    is_tensor_handle_feed = isinstance(subfeed_val, session_ops.TensorHandle)
                    if is_tensor_handle_feed:
                        np_val = subfeed_val.to_numpy_array()
                        feed_handles[subfeed_t.ref()] = subfeed_val
                    else:
                        np_val = np.asarray(subfeed_val, dtype=subfeed_dtype)
                    if not is_tensor_handle_feed and (not subfeed_t.get_shape().is_compatible_with(np_val.shape)):
                        raise ValueError(f'Cannot feed value of shape {str(np_val.shape)} for Tensor {subfeed_t.name}, which has shape {str(subfeed_t.get_shape())}')
                    if not self.graph.is_feedable(subfeed_t):
                        raise ValueError(f'Tensor {subfeed_t.name} may not be fed.')
                    feed_dict_tensor[subfeed_t.ref()] = np_val
                    feed_map[compat.as_bytes(subfeed_t.name)] = (subfeed_t, subfeed_val)
        fetch_handler = _FetchHandler(self._graph, fetches, feed_dict_tensor, feed_handles=feed_handles)
        _ = self._update_with_movers(feed_dict_tensor, feed_map)
        final_fetches = fetch_handler.fetches()
        final_targets = fetch_handler.targets()
        if final_fetches or final_targets or (handle and feed_dict_tensor):
            results = self._do_run(handle, final_targets, final_fetches, feed_dict_tensor, options, run_metadata)
        else:
            results = []
        return fetch_handler.build_results(self, results)

    def make_callable(self, fetches, feed_list=None, accept_options=False):
        """Returns a Python callable that runs a particular step.

    The returned callable will take `len(feed_list)` arguments whose types
    must be compatible feed values for the respective elements of `feed_list`.
    For example, if element `i` of `feed_list` is a `tf.Tensor`, the `i`th
    argument to the returned callable must be a numpy ndarray (or something
    convertible to an ndarray) with matching element type and shape. See
    `tf.Session.run` for details of the allowable feed key and value types.

    The returned callable will have the same return type as
    `tf.Session.run(fetches, ...)`. For example, if `fetches` is a `tf.Tensor`,
    the callable will return a numpy ndarray; if `fetches` is a `tf.Operation`,
    it will return `None`.

    Args:
      fetches: A value or list of values to fetch. See `tf.Session.run` for
        details of the allowable fetch types.
      feed_list: (Optional.) A list of `feed_dict` keys. See `tf.Session.run`
        for details of the allowable feed key types.
      accept_options: (Optional.) If `True`, the returned `Callable` will be
        able to accept `tf.compat.v1.RunOptions` and `tf.compat.v1.RunMetadata`
        as optional keyword arguments `options` and `run_metadata`,
        respectively, with the same syntax and semantics as `tf.Session.run`,
        which is useful for certain use cases (profiling and debugging) but will
        result in measurable slowdown of the `Callable`'s
        performance. Default: `False`.

    Returns:
      A function that when called will execute the step defined by
      `feed_list` and `fetches` in this session.

    Raises:
      TypeError: If `fetches` or `feed_list` cannot be interpreted
        as arguments to `tf.Session.run`.
    """
        if feed_list is not None:
            if not isinstance(feed_list, (list, tuple)):
                raise TypeError(f'Argument `feed_list` must be a list or tuple. Received: feed_list={feed_list}')

            def _generic_run(*feed_args, **kwargs):
                feed_dict = {feed: feed_val for feed, feed_val in zip(feed_list, feed_args)}
                return self.run(fetches, feed_dict=feed_dict, **kwargs)
            return _generic_run
        self._extend_graph()
        fetch_handler = _FetchHandler(self._graph, fetches, {})
        fetch_list = [t._as_tf_output() for t in fetch_handler.fetches()]
        target_list = [op._c_op for op in fetch_handler.targets()]

        def _callable_template_with_options_and_metadata(fetch_list, target_list, fetch_handler, options=None, run_metadata=None):
            """Template callable that accepts RunOptions and RunMetadata."""
            options_ptr = tf_session.TF_NewBufferFromString(compat.as_bytes(options.SerializeToString())) if options else None
            run_metadata_ptr = tf_session.TF_NewBuffer() if run_metadata else None
            try:
                results = self._call_tf_sessionrun(options_ptr, {}, fetch_list, target_list, run_metadata_ptr)
                if fetch_handler:
                    results = fetch_handler.build_results(self, results)
                else:
                    results = results[0] if results else None
                if run_metadata:
                    proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
                    run_metadata.ParseFromString(compat.as_bytes(proto_data))
            finally:
                if run_metadata_ptr:
                    tf_session.TF_DeleteBuffer(run_metadata_ptr)
                if options:
                    tf_session.TF_DeleteBuffer(options_ptr)
            return results
        if accept_options:
            return functools.partial(_callable_template_with_options_and_metadata, fetch_list, target_list, fetch_handler)
        elif isinstance(fetches, ops.Operation):
            assert not fetch_list
            assert len(target_list) == 1

            def _single_operation_run():
                self._call_tf_sessionrun(None, {}, [], target_list, None)
            return _single_operation_run
        elif isinstance(fetches, tensor.Tensor):
            assert len(fetch_list) == 1
            assert not target_list

            def _single_tensor_run():
                results = self._call_tf_sessionrun(None, {}, fetch_list, [], None)
                return results[0]
            return _single_tensor_run
        else:

            def _fetch_handler_run():
                results = self._call_tf_sessionrun(None, {}, fetch_list, target_list, None)
                return fetch_handler.build_results(self, results)
            return _fetch_handler_run
    _NODEDEF_NAME_RE = re.compile('\\[\\[(Node: )?(\\{\\{node )?([^\\} ]*)(\\}\\})?\\s*=*')

    def _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata):
        """Runs a step based on the given fetches and feeds.

    Args:
      handle: a handle for partial_run. None if this is just a call to run().
      target_list: A list of operations to be run, but not fetched.
      fetch_list: A list of tensors to be fetched.
      feed_dict: A dictionary that maps tensors to numpy ndarrays.
      options: A (pointer to a) [`RunOptions`] protocol buffer, or None
      run_metadata: A (pointer to a) [`RunMetadata`] protocol buffer, or None

    Returns:
      A list of numpy ndarrays, corresponding to the elements of
      `fetch_list`.  If the ith element of `fetch_list` contains the
      name of an operation, the first Tensor output of that operation
      will be returned for that element.

    Raises:
      tf.errors.OpError: Or one of its subclasses on error.
    """
        feeds = dict(((t.deref()._as_tf_output(), v) for t, v in feed_dict.items()))
        fetches = [t._as_tf_output() for t in fetch_list]
        targets = [op._c_op for op in target_list]

        def _run_fn(feed_dict, fetch_list, target_list, options, run_metadata):
            self._extend_graph()
            return self._call_tf_sessionrun(options, feed_dict, fetch_list, target_list, run_metadata)

        def _prun_fn(handle, feed_dict, fetch_list):
            if target_list:
                raise RuntimeError(f'partial_run() requires empty `target_list`. Received: target_list={target_list} (non-empty)')
            return self._call_tf_sessionprun(handle, feed_dict, fetch_list)
        if handle is None:
            return self._do_call(_run_fn, feeds, fetches, targets, options, run_metadata)
        else:
            return self._do_call(_prun_fn, handle, feeds, fetches)

    def _do_call(self, fn, *args):
        try:
            return fn(*args)
        except errors.OpError as e:
            message = compat.as_text(e.message)
            m = BaseSession._NODEDEF_NAME_RE.search(message)
            node_def = None
            op = None
            if m is not None:
                node_name = m.group(3)
                try:
                    op = self._graph.get_operation_by_name(node_name)
                    node_def = op.node_def
                except KeyError:
                    pass
            message = error_interpolation.interpolate_graph(message, self._graph)
            if 'only supports NHWC tensor format' in message:
                message += '\nA possible workaround: Try disabling Grappler optimizer\nby modifying the config for creating the session eg.\nsession_config.graph_options.rewrite_options.disable_meta_optimizer = True'
            raise type(e)(node_def, op, message)

    def _extend_graph(self):
        with self._graph._session_run_lock():
            tf_session.ExtendSession(self._session)
    _DEAD_HANDLES_THRESHOLD = 10

    def _register_dead_handle(self, handle):
        tensors_to_delete = None
        with self._delete_lock:
            self._dead_handles.append(handle)
            if len(self._dead_handles) == BaseSession._DEAD_HANDLES_THRESHOLD:
                tensors_to_delete = self._dead_handles
                self._dead_handles = []
        if tensors_to_delete:
            feeds = {}
            fetches = []
            for deleter_key, tensor_handle in enumerate(tensors_to_delete):
                holder, deleter = session_ops._get_handle_deleter(self.graph, deleter_key, tensor_handle)
                feeds[holder] = tensor_handle
                fetches.append(deleter)
            self.run(fetches, feed_dict=feeds)

    def _update_with_movers(self, feed_dict, feed_map):
        handle_movers = []
        for feed_name, val in feed_map.items():
            mover = session_ops._get_handle_mover(self.graph, *val)
            if mover:
                handle_movers.append((feed_name, val[1], mover))
        if not handle_movers:
            return []
        else:
            feeds = {}
            fetches = []
            for _, handle, mover in handle_movers:
                feeds[mover[0]] = handle
                fetches.append(mover[1])
            handles = self.run(fetches, feed_dict=feeds)
            for handle_mover, handle in zip(handle_movers, handles):
                np_val = np.array(handle.handle, dtype=np.object_)
                feed_name = handle_mover[0]
                feed_tensor = feed_map[feed_name][0]
                feed_dict[feed_tensor.ref()] = np_val
            return handles

    def _call_tf_sessionrun(self, options, feed_dict, fetch_list, target_list, run_metadata):
        return tf_session.TF_SessionRun_wrapper(self._session, options, feed_dict, fetch_list, target_list, run_metadata)

    def _call_tf_sessionprun(self, handle, feed_dict, fetch_list):
        return tf_session.TF_SessionPRun_wrapper(self._session, handle, feed_dict, fetch_list)

    class _Callable(object):
        """Experimental wrapper for the C++ `Session::MakeCallable()` API."""

        def __init__(self, session, callable_options):
            self._session = session
            self._handle = None
            options_ptr = tf_session.TF_NewBufferFromString(compat.as_bytes(callable_options.SerializeToString()))
            try:
                self._handle = tf_session.TF_SessionMakeCallable(session._session, options_ptr)
            finally:
                tf_session.TF_DeleteBuffer(options_ptr)

        def __call__(self, *args, **kwargs):
            run_metadata = kwargs.get('run_metadata', None)
            try:
                run_metadata_ptr = tf_session.TF_NewBuffer() if run_metadata else None
                ret = tf_session.TF_SessionRunCallable(self._session._session, self._handle, args, run_metadata_ptr)
                if run_metadata:
                    proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
                    run_metadata.ParseFromString(compat.as_bytes(proto_data))
            finally:
                if run_metadata_ptr:
                    tf_session.TF_DeleteBuffer(run_metadata_ptr)
            return ret

        def __del__(self):
            if self._handle is not None and self._session._session is not None and (not self._session._closed):
                tf_session.TF_SessionReleaseCallable(self._session._session, self._handle)

    def _make_callable_from_options(self, callable_options):
        """Returns a handle to a "callable" with the given options.

    Args:
      callable_options: A `CallableOptions` protocol buffer message describing
        the computation that will be performed by the callable.

    Returns:
      A handle to the new callable.
    """
        self._extend_graph()
        return BaseSession._Callable(self, callable_options)