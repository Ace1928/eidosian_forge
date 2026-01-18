from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _registered_kl(type_a, type_b):
    """Get the KL function registered for classes a and b."""
    hierarchy_a = tf_inspect.getmro(type_a)
    hierarchy_b = tf_inspect.getmro(type_b)
    dist_to_children = None
    kl_fn = None
    for mro_to_a, parent_a in enumerate(hierarchy_a):
        for mro_to_b, parent_b in enumerate(hierarchy_b):
            candidate_dist = mro_to_a + mro_to_b
            candidate_kl_fn = _DIVERGENCES.get((parent_a, parent_b), None)
            if not kl_fn or (candidate_kl_fn and candidate_dist < dist_to_children):
                dist_to_children = candidate_dist
                kl_fn = candidate_kl_fn
    return kl_fn