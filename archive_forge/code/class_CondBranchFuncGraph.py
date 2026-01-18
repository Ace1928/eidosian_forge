from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
class CondBranchFuncGraph(ControlFlowFuncGraph):
    """FuncGraph for branches of tf.cond().

  This is used to distinguish cond branches from other functions.
  """