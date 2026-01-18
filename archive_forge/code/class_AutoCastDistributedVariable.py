import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.types import core
class AutoCastDistributedVariable(AutoCastVariable, variable.__class__):
    """An AutoCastVariable that also subclasses from variable.__class__.

    variable.__class__ is either a DistributedVariable or an
    AggregatingVariable.
    """

    def __repr__(self):
        return '<AutoCastDistributedVariable dtype={v.dtype.name} dtype_to_cast_to={v._cast_dtype.name} inner_variable={v._variable}>'.format(v=self)