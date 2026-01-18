import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
def _registered_inverse(type_a):
    """Get the Cholesky function registered for class a."""
    return _registered_function([type_a], _INVERSES)