import numpy as np
from onnx.reference.ops.aionnx_preview_training._op_run_training import OpRunTraining
def _apply_adam(r, t, x, g, v, h, norm_coefficient, norm_coefficient_post, alpha, beta, epsilon):
    g_regularized = norm_coefficient * x + g
    v_new = alpha * v + (1 - alpha) * g_regularized
    h_new = beta * h + (1 - beta) * (g_regularized * g_regularized)
    h_sqrt = np.sqrt(h_new) + epsilon
    r_adjusted = None
    if t > 0:
        r_adjusted = r * np.sqrt(1 - beta ** t) / (1 - alpha ** t)
    else:
        r_adjusted = r
    x_new = x - r_adjusted * (v_new / h_sqrt)
    x_final = (1 - norm_coefficient_post) * x_new
    return (x_final, v_new, h_new)