import abc
import re
import threading
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import errors
from tensorflow.python.framework import stack
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import monitored_session
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
class BaseDebugWrapperSession(session.SessionInterface, metaclass=abc.ABCMeta):
    """Base class of debug-wrapper session classes.

  Concrete classes that inherit from this class need to implement the abstract
  methods such as on_session_init, on_run_start and on_run_end.
  """

    def __init__(self, sess, thread_name_filter=None, pass_through_operrors=False):
        """Constructor of `BaseDebugWrapperSession`.

    Args:
      sess: An (unwrapped) TensorFlow session instance. It should be a subtype
        of `BaseSession` or `tf.MonitoredSession`.
      thread_name_filter: Regular-expression filter (allowlist) for name(s) of
        thread(s) on which the wrapper session will be active. This regular
        expression is used in a start-anchored fashion on the thread name, i.e.,
        by applying the `match` method of the compiled pattern. The default
        `None` means that the wrapper session will be active on all threads.
        E.g., r"MainThread$", r"QueueRunnerThread.*".
      pass_through_operrors: If True, all captured OpErrors will be
        propagated.  By default this captures all OpErrors.

    Raises:
      ValueError: On invalid `OnSessionInitAction` value.
      NotImplementedError: If a non-DirectSession sess object is received.
    """
        _check_type(sess, (session.BaseSession, monitored_session.MonitoredSession))
        self._sess = sess
        self._thread_name_filter_pattern = re.compile(thread_name_filter) if thread_name_filter else None
        self._pass_through_operrors = pass_through_operrors
        self._run_call_count = 0
        response = self.on_session_init(OnSessionInitRequest(self._sess))
        _check_type(response, OnSessionInitResponse)
        if response.action == OnSessionInitAction.PROCEED:
            pass
        elif response.action == OnSessionInitAction.REMOTE_INSTR_LOOP:
            raise NotImplementedError('OnSessionInitAction REMOTE_INSTR_LOOP has not been implemented.')
        else:
            raise ValueError('Invalid OnSessionInitAction value: %s' % response.action)
        self._default_session_context_manager = None
        self._cached_callables_from_options = {}

    @property
    def graph(self):
        return self._sess.graph

    @property
    def graph_def(self):
        return self._sess.graph_def

    @property
    def sess_str(self):
        return self._sess.sess_str

    @property
    def session(self):
        return self._sess

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None, callable_runner=None, callable_runner_args=None, callable_options=None):
        """Wrapper around Session.run() that inserts tensor watch options.

    Args:
      fetches: Same as the `fetches` arg to regular `Session.run()`.
      feed_dict: Same as the `feed_dict` arg to regular `Session.run()`.
      options: Same as the `options` arg to regular `Session.run()`.
      run_metadata: Same as the `run_metadata` arg to regular `Session.run()`.
      callable_runner: A `callable` returned by `Session.make_callable()`.
        If not `None`, `fetches` and `feed_dict` must both be `None`.
        Mutually exclusive with `callable_options`.
      callable_runner_args: An optional list of arguments to `callable_runner`
        or for `callable_options`.
      callable_options: An instance of `config_pb2.CallableOptions`, to be
        used with `Session._make_callable_from_options()`. Mutually exclusive
        with `callable_runner`.

    Returns:
      Simply forwards the output of the wrapped `Session.run()` call.

    Raises:
      ValueError: On invalid `OnRunStartAction` value. Or if `callable_runner`
        is not `None` and either or both of `fetches` and `feed_dict` is `None`.
    """
        if callable_runner and callable_options:
            raise ValueError('callable_runner and callable_options are mutually exclusive, but are both specified in this call to BaseDebugWrapperSession.run().')
        if callable_runner and (fetches or feed_dict):
            raise ValueError('callable_runner and fetches/feed_dict are mutually exclusive, but are used simultaneously.')
        elif callable_options and (fetches or feed_dict):
            raise ValueError('callable_options and fetches/feed_dict are mutually exclusive, but are used simultaneously.')
        self.increment_run_call_count()

        def is_empty(x):
            """Check whether a possibly nested structure is empty."""
            if not nest.is_nested(x):
                return False
            if isinstance(x, collections_abc.Mapping):
                return is_empty(list(x.values()))
            for item in x:
                if not is_empty(item):
                    return False
            return True
        empty_fetches = is_empty(fetches)
        if empty_fetches:
            tf_logging.info('Due to empty fetches, tfdbg Session wrapper is letting a Session.run pass through without any debugging actions.')
        if self._is_disabled_thread() or empty_fetches:
            if callable_runner:
                return callable_runner(*callable_runner_args)
            elif callable_options:
                return self._sess._make_callable_from_options(callable_options)(*callable_runner_args)
            else:
                return self._sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
        run_start_resp = self.on_run_start(OnRunStartRequest(fetches, feed_dict, options, run_metadata, self._run_call_count, is_callable_runner=bool(callable_runner)))
        _check_type(run_start_resp, OnRunStartResponse)
        if run_start_resp.action == OnRunStartAction.DEBUG_RUN:
            retvals, run_end_req = self._run_with_debugging(run_start_resp, fetches, feed_dict, options, run_metadata, callable_runner, callable_runner_args, callable_options)
        elif run_start_resp.action == OnRunStartAction.PROFILE_RUN:
            retvals, run_end_req = self._run_with_profiling(run_start_resp, fetches, feed_dict, options, run_metadata, callable_runner, callable_runner_args, callable_options)
        elif run_start_resp.action == OnRunStartAction.NON_DEBUG_RUN:
            if callable_runner:
                retvals = callable_runner(*callable_runner_args)
            elif callable_options:
                callable_object = self._sess._make_callable_from_options(callable_options)
                retvals = callable_object(*callable_runner_args)
            else:
                retvals = self._sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
            run_end_req = OnRunEndRequest(run_start_resp.action)
        else:
            raise ValueError('Invalid OnRunStartAction value: %s' % run_start_resp.action)
        run_end_resp = self.on_run_end(run_end_req)
        _check_type(run_end_resp, OnRunEndResponse)
        return retvals

    def _run_with_debugging(self, run_start_resp, fetches, feed_dict, options, run_metadata, callable_runner, callable_runner_args, callable_options):
        """Perform a session.run() or callable with debugging."""
        decorated_run_options = None
        if callable_options:
            callable_options_id = id(callable_options)
            if callable_options_id not in self._cached_callables_from_options:
                new_callable_options = config_pb2.CallableOptions()
                new_callable_options.CopyFrom(callable_options)
                decorated_run_options = new_callable_options.run_options
        else:
            decorated_run_options = options or config_pb2.RunOptions()
        run_metadata = run_metadata or config_pb2.RunMetadata()
        if decorated_run_options:
            self._decorate_run_options_for_debug(decorated_run_options, run_start_resp.debug_urls, debug_ops=run_start_resp.debug_ops, node_name_regex_allowlist=run_start_resp.node_name_regex_allowlist, op_type_regex_allowlist=run_start_resp.op_type_regex_allowlist, tensor_dtype_regex_allowlist=run_start_resp.tensor_dtype_regex_allowlist, tolerate_debug_op_creation_failures=run_start_resp.tolerate_debug_op_creation_failures)
        tf_error = None
        try:
            if callable_runner:
                retvals = callable_runner(*callable_runner_args, options=decorated_run_options, run_metadata=run_metadata)
            elif callable_options:
                if callable_options_id in self._cached_callables_from_options:
                    callable_object = self._cached_callables_from_options[callable_options_id]
                else:
                    callable_object = self._sess._make_callable_from_options(new_callable_options)
                    self._cached_callables_from_options[callable_options_id] = callable_object
                retvals = callable_object(*callable_runner_args, run_metadata=run_metadata)
            else:
                retvals = self._sess.run(fetches, feed_dict=feed_dict, options=decorated_run_options, run_metadata=run_metadata)
        except errors.OpError as op_error:
            if self._pass_through_operrors:
                raise op_error
            tf_error = op_error
            retvals = op_error
        return (retvals, OnRunEndRequest(run_start_resp.action, run_metadata=run_metadata, client_graph_def=self._sess.graph.as_graph_def(), tf_error=tf_error))

    def _run_with_profiling(self, run_start_resp, fetches, feed_dict, options, run_metadata, callable_runner, callable_runner_args, callable_options):
        """Perform a session.run() or callable with profiling."""
        decorated_run_options = None
        if callable_options:
            callable_options_id = id(callable_options)
            if callable_options_id not in self._cached_callables_from_options:
                new_callable_options = config_pb2.CallableOptions()
                new_callable_options.CopyFrom(callable_options)
                decorated_run_options = new_callable_options.run_options
        else:
            decorated_run_options = options or config_pb2.RunOptions()
        self._decorate_run_options_for_profile(decorated_run_options)
        run_metadata = run_metadata or config_pb2.RunMetadata()
        if callable_runner:
            retvals = callable_runner(*callable_runner_args, options=decorated_run_options, run_metadata=run_metadata)
        elif callable_options:
            callable_object = self._sess._make_callable_from_options(new_callable_options)
            retvals = callable_object(*callable_runner_args, run_metadata=run_metadata)
        else:
            retvals = self._sess.run(fetches, feed_dict=feed_dict, options=decorated_run_options, run_metadata=run_metadata)
        return (retvals, OnRunEndRequest(run_start_resp.action, run_metadata=run_metadata, client_graph_def=self._sess.graph.as_graph_def()))

    def _is_disabled_thread(self):
        thread_name = threading.current_thread().name or ''
        return self._thread_name_filter_pattern and (not self._thread_name_filter_pattern.match(thread_name))

    def run_step_fn(self, step_fn):
        return step_fn(monitored_session.MonitoredSession.StepContext(self._sess, self.run))

    def partial_run_setup(self, fetches, feeds=None):
        """Sets up the feeds and fetches for partial runs in the session."""
        raise NotImplementedError('partial_run_setup is not implemented for debug-wrapper sessions.')

    def partial_run(self, handle, fetches, feed_dict=None):
        raise NotImplementedError('partial_run is not implemented for debug-wrapper sessions.')

    def list_devices(self, *args, **kwargs):
        return self._sess.list_devices(*args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._sess.reset(*args, **kwargs)

    def make_callable(self, fetches, feed_list=None, accept_options=False):
        runner = self._sess.make_callable(fetches, feed_list=feed_list, accept_options=True)

        def wrapped_runner(*runner_args, **kwargs):
            return self.run(None, feed_dict=None, options=kwargs.get('options', None), run_metadata=kwargs.get('run_metadata', None), callable_runner=runner, callable_runner_args=runner_args)
        return wrapped_runner

    def _make_callable_from_options(self, callable_options):

        def wrapped_runner(*feed_values, **kwargs):
            return self.run(None, run_metadata=kwargs.get('run_metadata', None), callable_options=callable_options, callable_runner_args=feed_values)
        return wrapped_runner

    @property
    def run_call_count(self):
        return self._run_call_count

    def increment_run_call_count(self):
        self._run_call_count += 1

    def _is_disk_usage_reset_each_run(self):
        """Indicates whether disk usage is reset after each Session.run.

    Subclasses that clean up the disk usage after every run should
    override this protected method.

    Returns:
      (`bool`) Whether the disk usage amount is reset to zero after
        each Session.run.
    """
        return False

    def _decorate_run_options_for_debug(self, run_options, debug_urls, debug_ops='DebugIdentity', node_name_regex_allowlist=None, op_type_regex_allowlist=None, tensor_dtype_regex_allowlist=None, tolerate_debug_op_creation_failures=False):
        """Modify a RunOptions object for debug tensor watching.

    Specifies request for outputting partition graphs. Adds
    debug_tensor_watch_opts with proper debug URLs.

    Args:
      run_options: (RunOptions) the modified RunOptions object.
      debug_urls: (list of str) debug URLs to be entered in run_options.
        debug_tensor_watch_opts.
      debug_ops: (str or list of str) debug op(s) to be used by the debugger.
      node_name_regex_allowlist: Regular-expression allowlist for node
        name.
      op_type_regex_allowlist: Regular-expression allowlist for op type.
      tensor_dtype_regex_allowlist: Regular-expression allowlist for tensor
        dtype.
      tolerate_debug_op_creation_failures: Whether debug op creation failures
        are to be tolerated.
    """
        run_options.output_partition_graphs = True
        debug_utils.watch_graph(run_options, self._sess.graph, debug_urls=debug_urls, debug_ops=debug_ops, node_name_regex_allowlist=node_name_regex_allowlist, op_type_regex_allowlist=op_type_regex_allowlist, tensor_dtype_regex_allowlist=tensor_dtype_regex_allowlist, tolerate_debug_op_creation_failures=tolerate_debug_op_creation_failures, reset_disk_byte_usage=self._run_call_count == 1 or self._is_disk_usage_reset_each_run())

    def _decorate_run_options_for_profile(self, run_options):
        """Modify a RunOptions object for profiling TensorFlow graph execution.

    Args:
      run_options: (RunOptions) the modified RunOptions object.
    """
        run_options.trace_level = config_pb2.RunOptions.FULL_TRACE

    @abc.abstractmethod
    def on_session_init(self, request):
        """Callback invoked during construction of the debug-wrapper session.

    This is a blocking callback.
    The invocation happens right before the constructor ends.

    Args:
      request: (`OnSessionInitRequest`) callback request carrying information
        such as the session being wrapped.

    Returns:
      An instance of `OnSessionInitResponse`.
    """

    @abc.abstractmethod
    def on_run_start(self, request):
        """Callback invoked on run() calls to the debug-wrapper session.

    This is a blocking callback.
    The invocation happens after the wrapper's run() call is entered,
    after an increment of run call counter.

    Args:
      request: (`OnRunStartRequest`) callback request object carrying
        information about the run call such as the fetches, feed dict, run
        options, run metadata, and how many `run()` calls to this wrapper
        session have occurred.

    Returns:
      An instance of `OnRunStartResponse`, carrying information to
        debug URLs used to watch the tensors.
    """

    @abc.abstractmethod
    def on_run_end(self, request):
        """Callback invoked on run() calls to the debug-wrapper session.

    This is a blocking callback.
    The invocation happens right before the wrapper exits its run() call.

    Args:
      request: (`OnRunEndRequest`) callback request object carrying information
        such as the actual action performed by the session wrapper for the
        run() call.

    Returns:
      An instance of `OnRunStartResponse`.
    """

    def as_default(self):
        return stack.default_session(self)

    def __enter__(self):
        if self._default_session_context_manager is None:
            self._default_session_context_manager = self.as_default()
        return self._default_session_context_manager.__enter__()

    def __exit__(self, exec_type, exec_value, exec_tb):
        self._default_session_context_manager.__exit__(exec_type, exec_value, exec_tb)

    def __del__(self):
        if hasattr(self._sess, '__del__'):
            self._sess.__del__()

    def close(self):
        self._sess.close()

    def should_stop(self):
        if hasattr(self._sess, 'should_stop'):
            return self._sess.should_stop()
        else:
            raise ValueError("The wrapped session %r does not have a method called 'should_stop'. Do you intend to wrap a tf.MonitoredSession instead?" % self._sess)