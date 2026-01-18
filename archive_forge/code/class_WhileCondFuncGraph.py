from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
class WhileCondFuncGraph(ControlFlowFuncGraph):
    """FuncGraph for the condition of tf.while_loop().

  This is used to distinguish while conditions from other functions.
  """