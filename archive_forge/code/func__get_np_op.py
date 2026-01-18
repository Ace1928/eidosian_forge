import numpy as onp
def _get_np_op(name):
    """Get official NumPy operator with `name`. If not found, raise ValueError."""
    for mod in _ONP_OP_MODULES:
        op = getattr(mod, name, None)
        if op is not None:
            return op
    raise ValueError('Operator `{}` is not supported by `mxnet.numpy`.'.format(name))