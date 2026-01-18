import numpy as np
from onnx.reference.op_run import OpRun
def _conv_implementation_im2col(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides):
    if dilations is None:
        dilations = [1 for s in X.shape[2:]]
    if kernel_shape is None:
        kernel_shape = W.shape[2:]
    if pads is None:
        pads = [0 for s in X.shape[2:]] * 2
    if strides is None:
        strides = [1 for s in X.shape[2:]]
    kernel_shape = tuple(kernel_shape)
    if X.shape[1] != W.shape[1] * group or W.shape[0] % group != 0:
        raise ValueError(f'Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={group}, W should be {(W.shape[0], X.shape[1] // group, np.prod(W.shape[1:]) // X.shape[1] * group)}.')
    if group > 1:
        res = []
        td = 0
        mg = W.shape[0] // group
        dw = W.shape[1]
        for b in range(X.shape[0]):
            for g in range(group):
                gx = X[b:b + 1, g * dw:(g + 1) * dw]
                gw = W[g * mg:(g + 1) * mg]
                try:
                    cv = _conv_implementation_im2col(gx, gw, None, auto_pad, dilations, 1, kernel_shape, pads, strides)
                except (ValueError, RuntimeError) as e:
                    raise ValueError(f'Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={g}/{group}, gx.shape={gx.shape}, gw.shape={gw.shape}, auto_pad={auto_pad}, dilations={dilations}, kernel_shape={kernel_shape}, pads={pads}, strides={strides}.') from e
                if b == 0:
                    td += cv.shape[1]
                res.append((b, cv))
        new_shape = [X.shape[0], *list(res[0][1].shape[1:])]
        new_shape[1] = td
        final = np.zeros(tuple(new_shape), dtype=res[0][1].dtype)
        p = 0
        for b, cv in res:
            final[b:b + 1, p:p + cv.shape[1]] = cv
            p += cv.shape[1]
            if p >= final.shape[1]:
                p = 0
        if B is not None:
            new_shape = [1 for s in final.shape]
            new_shape[1] = B.shape[0]
            b = B.reshape(tuple(new_shape))
            final += b
        return final
    if dilations[0] != 1 or min(dilations) != max(dilations):
        nd = len(dilations)
        new_kernel_shape = []
        new_shape = list(W.shape[:-nd])
        for i, d in enumerate(dilations):
            di = len(W.shape) - nd + i
            new_shape.append(W.shape[di] + (W.shape[di] - 1) * (d - 1))
            new_kernel_shape.append(kernel_shape[i] + (kernel_shape[i] - 1) * (d - 1))
        new_w = np.zeros(tuple(new_shape), dtype=W.dtype)
        indices = [slice(0, new_w.shape[0]), slice(0, new_w.shape[1])]
        for i, d in enumerate(dilations):
            di = len(W.shape) - nd + i
            indices.append(slice(0, new_w.shape[di], d))
        new_w[tuple(indices)] = W
        W = new_w
        kernel_shape = new_kernel_shape
    if auto_pad in {'SAME_LOWER', 'SAME_UPPER', 'VALID'}:
        head = []
        tail = []
        for i in range(len(X.shape) - 2):
            d = X.shape[i]
            target_size = (d + strides[i] - 1) // strides[i]
            pad_needed = (target_size - 1) * strides[i] + kernel_shape[i] - d
            if auto_pad == 'SAME_LOWER':
                pad_head = (pad_needed + 1) // 2
            else:
                pad_head = pad_needed // 2
            pad_tail = pad_needed - pad_head
            head.append(pad_head)
            tail.append(pad_tail)
        pads = head + tail
    c2, out_shape = im2col_fast(X, kernel_shape, pads, strides)
    w_reshaped = W.reshape((-1, c2.shape[0]))
    mul = w_reshaped @ c2
    mul = mul.reshape((W.shape[0], X.shape[0], *out_shape))
    perm = (1, 0, *tuple(np.arange(len(X.shape) - 2) + 2))
    mul = mul.transpose(perm)
    if B is not None:
        if B.size == 1:
            return mul + B
        new_shape = [1] * len(mul.shape)
        new_shape[1] = -1
        mul += B.reshape(tuple(new_shape))
    return mul