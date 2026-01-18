from cupy._core import _fusion_variable
from cupy._core import _fusion_op
def _fuse_consecutive_ops(ops, shape_constraints):
    res = []
    for op in ops:
        if len(res) == 0:
            res.append(op)
        else:
            prev_op = res.pop(-1)
            new_op = _fuse_two_ops(prev_op, op)
            if new_op is None:
                res.extend([prev_op, op])
            else:
                res.append(new_op)
    return res