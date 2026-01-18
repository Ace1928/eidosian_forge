import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def _is_math_fn(fn):
    mod = inspect.getmodule(fn)
    if not mod:
        raise RuntimeError(f'Module for {fn} not found')
    return mod.__name__ == 'math'