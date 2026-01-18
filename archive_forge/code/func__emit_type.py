import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def _emit_type(type):
    return str(type)