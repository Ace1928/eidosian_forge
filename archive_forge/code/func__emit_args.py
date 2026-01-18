import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def _emit_args(indent, arguments):
    return ','.join((_emit_arg(indent, i, arg) for i, arg in enumerate(arguments)))