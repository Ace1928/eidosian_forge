from onnx.reference.ops.aionnx_preview_training._op_run_training import OpRunTraining
def _apply_nesterov(r, t, x, g, v, norm_coefficient, alpha, beta):
    g_regularized = norm_coefficient * x + g
    beta_adjusted = beta if t > 0 else 1
    v_new = alpha * v + beta_adjusted * g_regularized
    x_new = x - r * (g_regularized + alpha * v_new)
    return (x_new, v_new)