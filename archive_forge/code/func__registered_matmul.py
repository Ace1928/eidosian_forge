import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
def _registered_matmul(type_a, type_b):
    """Get the Matmul function registered for classes a and b."""
    return _registered_function([type_a, type_b], _MATMUL)