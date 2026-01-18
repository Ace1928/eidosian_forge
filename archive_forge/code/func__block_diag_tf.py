from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _block_diag_tf(tensors):
    """TensorFlow implementation of scipy.linalg.block_diag"""
    tf = _i('tf')
    int_dtype = None
    if tensors[0].dtype in (tf.int32, tf.int64):
        int_dtype = tensors[0].dtype
        tensors = [tf.cast(t, tf.float32) for t in tensors]
    linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in tensors]
    linop_block_diagonal = tf.linalg.LinearOperatorBlockDiag(linop_blocks)
    res = linop_block_diagonal.to_dense()
    if int_dtype is None:
        return res
    return tf.cast(res, int_dtype)