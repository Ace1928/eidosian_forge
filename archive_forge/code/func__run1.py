from onnx.reference.ops.aionnx_preview_training._op_run_training import OpRunTraining
def _run1(self, r, t, x, g, v, mode='standard', norm_coefficient=None, alpha=None, beta=None):
    if mode == 'standard':
        x_new, v_new = _apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta)
    else:
        x_new, v_new = _apply_nesterov(r, t, x, g, v, norm_coefficient, alpha, beta)
    return (x_new, v_new)