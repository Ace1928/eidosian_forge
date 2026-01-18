from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.operators import variables
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import nest
def _mark_return_if_tensor(t):
    if tensor_util.is_tf_type(t):
        return self.autodeps_scope.mark_as_return(t)
    return t