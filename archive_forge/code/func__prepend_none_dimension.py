from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_config
from tensorflow.python.ops.gen_parsing_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _prepend_none_dimension(features):
    """Returns a copy of features with adjusted FixedLenSequenceFeature shapes."""
    if features:
        modified_features = dict(features)
        for key, feature in features.items():
            if isinstance(feature, FixedLenSequenceFeature):
                if not feature.allow_missing:
                    raise ValueError('Unsupported: FixedLenSequenceFeature requires allow_missing to be True.')
                modified_features[key] = FixedLenSequenceFeature([None] + list(feature.shape), feature.dtype, feature.allow_missing, feature.default_value)
        return modified_features
    else:
        return features