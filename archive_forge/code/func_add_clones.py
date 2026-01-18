import torch
from .make_fx import make_fx_check
from .aot_autograd import aot_autograd_check
from .fake_tensor import fake_check
from torch._subclasses.schema_check_mode import SchemaCheckMode
def add_clones(func):

    def clone_tensors(args, kwargs):
        args = tuple((clone_arg(a) for a in args))
        kwargs = {k: clone_arg(v) for k, v in kwargs.items()}
        return (args, kwargs)

    def clone_then_func(*args, **kwargs):
        args, kwargs = clone_tensors(args, kwargs)
        return func(*args, **kwargs)
    return clone_then_func