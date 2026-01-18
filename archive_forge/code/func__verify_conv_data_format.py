import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
def _verify_conv_data_format(node):
    """Verifies data format for pooling and convolutional operations."""
    if node.attr['data_format'].s != b'NHWC':
        raise ValueError('Only NHWC format is supported in flops computations')