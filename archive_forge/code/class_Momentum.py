from onnx.reference.ops.aionnx_preview_training._op_run_training import OpRunTraining
class Momentum(OpRunTraining):

    def _run(self, *data, alpha=None, beta=None, mode=None, norm_coefficient=None):
        if len(data) == 5:
            r, t, x, g, v = data
            return self._run1(r, t, x, g, v, norm_coefficient=norm_coefficient, alpha=alpha, beta=beta, mode=mode)
        n = (len(data) - 2) // 3
        xs = []
        vs = []
        for i in range(0, n):
            a, b = self._run1(*data[:2], data[2 + i], data[2 + n + i], data[2 + n * 2 + i], norm_coefficient=norm_coefficient, alpha=alpha, beta=beta, mode=mode)
            xs.append(a)
            vs.append(b)
        return tuple(xs + vs)

    def _run1(self, r, t, x, g, v, mode='standard', norm_coefficient=None, alpha=None, beta=None):
        if mode == 'standard':
            x_new, v_new = _apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta)
        else:
            x_new, v_new = _apply_nesterov(r, t, x, g, v, norm_coefficient, alpha, beta)
        return (x_new, v_new)