import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def _hidden(name):
    return name.startswith('_') and (not name.startswith('__'))