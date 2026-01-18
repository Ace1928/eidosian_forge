from typing import Tuple
import numpy as np
from onnx.reference.op_run import OpRun
def _layer_normalization(X: np.ndarray, W: np.ndarray, B: np.ndarray, axis: int=-1, epsilon: float=1e-05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_shape = X.shape
    X_rank = len(X_shape)
    if axis < 0:
        axis = axis + X_rank
    unsqueezed_rank = X_rank - axis
    reduction_shape = X_shape[0:axis] + (1,) * unsqueezed_rank
    row_number = 1
    col_number = 1
    for i in range(X_rank):
        if i < axis:
            row_number *= X_shape[i]
        else:
            col_number *= X_shape[i]
    x_mat = np.reshape(X, (row_number, col_number))
    x_mean = np.sum(x_mat, axis=1, keepdims=True) / col_number
    x_diff = x_mat - x_mean
    x_squared_diff = x_diff * x_diff
    variance = np.sum(x_squared_diff, axis=1, keepdims=True) / col_number
    variance_eps = variance + epsilon
    std_dev = np.sqrt(variance_eps)
    inv_std_dev = np.reciprocal(std_dev)
    y_mat = x_diff * inv_std_dev
    Y = np.reshape(y_mat, X_shape) * W
    if B is not None:
        Y = Y + B
    X_mean = np.reshape(x_mean, reduction_shape)
    X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)
    return (Y.astype(X.dtype), X_mean.astype(X.dtype), X_inv_std_dev.astype(X.dtype))