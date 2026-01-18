import warnings
import cupy
from cupy._core import _accelerator
def _try_use_cutensornet(*args, **kwargs):
    if cupy.cuda.runtime.is_hip:
        return None
    if _accelerator.ACCELERATOR_CUTENSORNET not in _accelerator.get_routine_accelerators():
        return None
    if cutensornet is None:
        warnings.warn('using the cuTensorNet backend was requested but it cannot be imported -- maybe you forgot to install cuQuantum Python? Please do "pip install cuquantum-python" or "conda install -c conda-forge cuquantum-python" and retry', stacklevel=2)
        return None
    dtype = kwargs.get('dtype', None)
    path = kwargs.get('optimize', False)
    if path is True:
        path = 'greedy'
    args = _get_einsum_operands(args)
    operands = [cupy.asarray(op) for op in args[1]]
    if len(operands) == 1:
        return None
    if any((op.size == 0 for op in operands)) or any((len(op.shape) == 0 for op in operands)):
        return None
    result_dtype = cupy.result_type(*operands) if dtype is None else dtype
    if result_dtype not in (cupy.float32, cupy.float64, cupy.complex64, cupy.complex128):
        return None
    operands = [op.astype(result_dtype, copy=False) for op in operands]
    device = cupy.cuda.runtime.getDevice()
    handle = cutn_handle_cache.get(device)
    if handle is None:
        handle = cutensornet.create()
        cutn_handle_cache[device] = handle
    cutn_options = {'device_id': device, 'handle': handle}
    raise_warning = False
    if path is False:
        path = [(i - 1, i - 2) for i in range(len(operands), 1, -1)]
    elif len(path) and path[0] == 'einsum_path':
        path = path[1:]
    elif len(path) == 2:
        if isinstance(path[1], (int, float)):
            raise_warning = True
        if path[0] != 'cutensornet':
            raise_warning = True
        path = None
    else:
        if path != 'cutensornet':
            raise_warning = True
        path = None
    if raise_warning:
        warnings.warn('the cuTensorNet backend ignores the "optimize" option except when an explicit contraction path is provided or when optimize=False (disable optimization); also, the maximum intermediate size, if set, is ignored', stacklevel=2)
    cutn_optimizer = {'path': path} if path else None
    if len(args) == 2:
        out = cutensornet.contract(args[0], *operands, options=cutn_options, optimize=cutn_optimizer)
    elif len(args) == 3:
        inputs = [i for pair in zip(operands, args[0]) for i in pair]
        if args[2] is not None:
            inputs.append(args[2])
        out = cutensornet.contract(*inputs, options=cutn_options, optimize=cutn_optimizer)
    else:
        assert False
    return out