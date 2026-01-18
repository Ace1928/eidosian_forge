from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
class _GradLoopState:
    """The state used for constructing the gradient graph for a while loop.

  We create a _GradLoopState for each while loop in forward and its
  corresponding while loop in backprop. This gives us access to both
  the forward and the backprop WhileContexts.

  During the construction of gradient graph, any time when we detect
  a forward value that is needed for backprop, we create a history
  accumulator and add it to `history_map`. Any time when we backprop
  a loop switch op (in _SwitchGrad), we add the grad merge op in
  `switch_map`.
  """

    def __init__(self, forward_ctxt, outer_grad_state):
        self._outer_grad_state = None
        self._forward_context = None
        self._forward_index = None
        self._forward_sync = None
        self._grad_context = None
        self._grad_index = None
        self._grad_sync = None
        self._history_map = {}
        self._switch_map = {}
        self._unused_exits = []
        self._deferred_exits = []
        self._forward_loop_exits = list(forward_ctxt.loop_exits)
        self._pending_exits_count = len(forward_ctxt.loop_exits)
        self._outer_grad_state = outer_grad_state
        if outer_grad_state:
            outer_forward_ctxt = outer_grad_state.forward_context
        else:
            if not hasattr(forward_ctxt, 'outer_context'):
                raise ValueError('Failed to call gradients on a while loop withoutproperly serializing graph via MetaGraphDef')
            outer_forward_ctxt = forward_ctxt.outer_context
        with forward_ctxt._graph.as_default():
            if outer_forward_ctxt:
                outer_forward_ctxt.Enter()
            cnt, forward_index = forward_ctxt.AddForwardLoopCounter(outer_grad_state)
            if outer_forward_ctxt:
                outer_forward_ctxt.Exit()
        self._forward_context = forward_ctxt
        self._forward_index = forward_index
        if outer_grad_state:
            outer_forward_ctxt.AddName(cnt.name)
            history_cnt = outer_grad_state.AddForwardAccumulator(cnt)
            outer_grad_ctxt = outer_grad_state.grad_context
            outer_grad_ctxt.Enter()
            self._grad_context = control_flow_ops.WhileContext(maximum_iterations=forward_ctxt.maximum_iterations, parallel_iterations=forward_ctxt.parallel_iterations, back_prop=forward_ctxt.back_prop, swap_memory=forward_ctxt.swap_memory, name=forward_ctxt.name, grad_state=self)
            real_cnt = outer_grad_state.AddBackpropAccumulatedValue(history_cnt, cnt)
            self._grad_index = self._grad_context.AddBackpropLoopCounter(real_cnt, outer_grad_state)
            outer_grad_ctxt.Exit()
        else:
            if outer_forward_ctxt:
                outer_forward_ctxt.Enter()
            self._grad_context = control_flow_ops.WhileContext(maximum_iterations=forward_ctxt.maximum_iterations, parallel_iterations=forward_ctxt.parallel_iterations, back_prop=forward_ctxt.back_prop, swap_memory=forward_ctxt.swap_memory, name=forward_ctxt.name, grad_state=self)
            self._grad_index = self._grad_context.AddBackpropLoopCounter(cnt, outer_grad_state)
            if outer_forward_ctxt:
                outer_forward_ctxt.Exit()

    @property
    def outer_grad_state(self):
        """The grad loop state for outer loop."""
        return self._outer_grad_state

    @property
    def forward_context(self):
        """The while loop context for forward."""
        return self._forward_context

    @property
    def forward_index(self):
        """The loop index of forward loop."""
        return self._forward_index

    @property
    def forward_sync(self):
        """A control trigger node for synchronization in the forward loop.

    One main use is to keep the push ops of a stack executed in the
    iteration order.
    """
        if self._forward_sync is None:
            with ops.control_dependencies(None):
                self._forward_sync = control_flow_ops.control_trigger(name='f_sync')
            self._forward_sync._set_control_flow_context(self._forward_context)
            self._forward_index.op._add_control_input(self._forward_sync)
        return self._forward_sync

    @property
    def grad_context(self):
        """The corresponding WhileContext for gradient."""
        return self._grad_context

    @property
    def grad_index(self):
        """The loop index of backprop loop."""
        return self._grad_index

    @property
    def grad_sync(self):
        """A control trigger node for synchronization in the grad loop.

    One main use is to keep the pop ops of a stack executed in the
    iteration order.
    """
        if self._grad_sync is None:
            with ops.control_dependencies(None):
                self._grad_sync = control_flow_ops.control_trigger(name='b_sync')
            self._grad_sync._set_control_flow_context(self._grad_context)
            self._grad_index.op._add_control_input(self._grad_sync)
            if self._grad_context.outer_context:
                self._grad_context.outer_context.AddInnerOp(self._grad_sync)
        return self._grad_sync

    @property
    def history_map(self):
        """The map that records all the tensors needed for backprop."""
        return self._history_map

    @property
    def switch_map(self):
        """The map that records all the Switch ops for the while loop."""
        return self._switch_map

    @property
    def unused_exits(self):
        """The list of "unused" exits."""
        return self._unused_exits

    @property
    def deferred_exits(self):
        """The list of "deferred" exits."""
        return self._deferred_exits

    @property
    def forward_loop_exits(self):
        """The list of exits of the forward loop."""
        return self._forward_loop_exits

    @property
    def pending_exits_count(self):
        """The number of exits we expect to see but haven't."""
        return self._pending_exits_count

    @pending_exits_count.setter
    def pending_exits_count(self, cnt):
        """Set the pending count to cnt."""
        self._pending_exits_count = cnt

    def AddForwardAccumulator(self, value, dead_branch=False):
        """Add an accumulator for each forward tensor that is needed in backprop.

    This is added to the forward loop at the first time when a tensor
    in the forward loop is used by backprop gradient computation loop.
    We create an accumulator that accumulates the value of tensor at each
    iteration. Called in the control flow context where gradients() is called.

    The pseudocode is:
    ```
      acc = stack();
      while (_pivot) {
        acc = stack_push(acc, value);
      }
    ```

    We make sure that the stack push op in one iteration is executed before
    next iteration. This is achieved by adding a control edge from
    `forward_index.op.inputs[0].op` to the push op, and another control
    edge from the push op to either `forward_index.op` or `forward_sync`.

    Args:
      value: The source tensor in forward that is to be accumulated.
      dead_branch: True iff the tensor is on a dead branch of a cond.

    Returns:
      The stack that contains the accumulated history of the tensor.

    Raises:
      TypeError: For internal errors involving the value condition context.
      ValueError: If `value` is inside a XLA scope and a valid max size
        for the stack can't be found.
    """
        with self._forward_index.graph.as_default():
            curr_ctxt = ops.get_default_graph()._get_control_flow_context()
            with ops.control_dependencies(None):
                if curr_ctxt:
                    curr_ctxt.Enter()
                with ops.colocate_with(value):
                    if not util.IsInXLAContext(value.op):
                        max_size = constant_op.constant(-1, dtypes.int32)
                    else:
                        max_size = _GetMaxSizeFromNestedMaximumIterations(value, self.forward_context)
                    acc = gen_data_flow_ops.stack_v2(max_size=max_size, elem_type=value.dtype.base_dtype, name='f_acc')
                if curr_ctxt:
                    curr_ctxt.Exit()
                enter_acc = self.forward_context.AddValue(acc)
                swap_enabled = self.forward_context.swap_memory
                value_ctxt = util.GetOutputContext(value.op)
                if value_ctxt == self.forward_context:
                    self.forward_context.Enter()
                    push = gen_data_flow_ops.stack_push_v2(enter_acc, value, swap_memory=swap_enabled)
                    self.forward_context.Exit()
                    self.forward_index.op._add_control_input(push.op)
                else:
                    if not isinstance(value_ctxt, control_flow_ops.CondContext):
                        raise TypeError('value_ctxt is not a CondContext: %s' % value_ctxt)
                    if dead_branch:
                        value_ctxt.outer_context.Enter()
                        push = gen_data_flow_ops.stack_push_v2(enter_acc, value, swap_memory=swap_enabled)
                        value_ctxt.outer_context.Exit()
                        push.op._set_control_flow_context(value_ctxt)
                    else:
                        value_ctxt.Enter()
                        push = gen_data_flow_ops.stack_push_v2(enter_acc, value, swap_memory=swap_enabled)
                        value_ctxt.Exit()
                    self.forward_sync._add_control_input(push.op)
                add_op = self.forward_index.op.inputs[0].op
                push.op._add_control_input(add_op)
                return acc

    def AddBackpropAccumulatedValue(self, history_value, value, dead_branch=False):
        """Add the getter for an accumulated value in the grad context.

    This is added to the backprop loop. Called in the grad context to
    get the value of an accumulated value. The stack pop op must be guarded
    by the pred of the controlling cond.

    Args:
      history_value: The history (a stack) of a value.
      value: The value that is pushed onto the stack.
      dead_branch: True iff the tensor is on a dead branch of a cond.

    Returns:
      The current value (the top of the stack).
    """
        history_ctxt = history_value.op._get_control_flow_context()
        cond_ctxt = None
        value_ctxt = value.op._get_control_flow_context()
        while value_ctxt and value_ctxt != history_ctxt:
            if isinstance(value_ctxt, control_flow_ops.CondContext):
                cond_ctxt = value_ctxt
                break
            value_ctxt = value_ctxt.outer_context
        with ops.control_dependencies(None):
            self.grad_context.Enter()
            if cond_ctxt:
                grad_state = self
                pred = None
                while pred is None and grad_state:
                    pred = grad_state.history_map.get(cond_ctxt.pred.name)
                    grad_state = grad_state.outer_grad_state
                if pred is None:
                    pred = cond_ctxt.pred
                branch = 1 - cond_ctxt.branch if dead_branch else cond_ctxt.branch
                history_value = control_flow_ops._SwitchRefOrTensor(history_value, pred)[branch]
            pop = gen_data_flow_ops.stack_pop_v2(history_value, value.dtype.base_dtype)
            pop.set_shape(value.get_shape())
            self.grad_context.Exit()
        parallel_iterations = self.grad_context.parallel_iterations
        if parallel_iterations > 1:
            self.grad_sync._add_control_input(pop.op)
        return pop

    def GetRealValue(self, value):
        """Get the real value of `value`.

    If backprop "uses" a value produced by forward inference, an accumulator
    is added in the forward loop to accumulate its values.  We use the
    accumulated value. This method must be called in the grad loop context.
    `value` must be in forward and needed for backprop.

    Args:
      value: A tensor to be captured.

    Returns:
      The same tensor obtained from the saved history.
    """
        assert value.op.type not in ['Variable', 'VariableV2']
        real_value = self._history_map.get(value.name)
        if real_value is None:
            cur_value = value
            cur_grad_state = self
            while True:
                enter_op = util.GetLoopConstantEnter(cur_value)
                if enter_op:
                    cur_value = enter_op.inputs[0]
                    cur_grad_state = cur_grad_state.outer_grad_state
                    if cur_grad_state is None:
                        real_value = self._grad_context.AddValue(cur_value)
                        break
                elif constant_op.is_constant(cur_value):
                    real_value = constant_op.constant(tensor_util.constant_value(cur_value), dtype=cur_value.dtype)
                    break
                else:
                    self._grad_context.Exit()
                    history_value = cur_grad_state.AddForwardAccumulator(cur_value)
                    self._grad_context.Enter()
                    break
            if real_value is None:
                real_value = cur_grad_state.AddBackpropAccumulatedValue(history_value, cur_value)
                if cur_grad_state != self:
                    real_value = self._grad_context.AddValue(real_value)
            self._history_map[value.name] = real_value
        return real_value