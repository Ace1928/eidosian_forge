from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _cond_tf(pred, true_fn, false_fn, args):
    import tensorflow as tf
    return tf.cond(pred, lambda: true_fn(*args), lambda: false_fn(*args))