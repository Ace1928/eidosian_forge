import math
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import math_ops
def _check_penalty_number(x):
    """check penalty number availability, raise ValueError if failed."""
    if not isinstance(x, (float, int)):
        raise ValueError('Value: {} is not a valid regularization penalty number, expected an int or float value'.format(x))
    if math.isinf(x) or math.isnan(x):
        raise ValueError('Value: {} is not a valid regularization penalty number, a positive/negative infinity or NaN is not a property value'.format(x))