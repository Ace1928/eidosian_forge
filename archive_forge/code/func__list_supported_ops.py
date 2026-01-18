import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def _list_supported_ops():

    def emit_block(decls):
        return '\n.. rst-class:: codeblock-height-limiter\n\n::\n\n{}\n'.format(''.join((f'    {d}\n\n' for d in decls)))
    body = ''
    op_gathering_fns = (_get_tensor_ops, _get_nn_functional_ops, _get_torchscript_builtins, _get_global_builtins, _get_math_builtins)
    for fn in op_gathering_fns:
        header, items = fn()
        link_target = header.replace('`', '').replace('-', '').lower().replace(' ', '-')
        if isinstance(items, str):
            section = f'{header}\n{'~' * len(header)}\n{items}\n'
        else:
            section = f'{header}\n{'~' * len(header)}\n{emit_block(items)}'
        section = f'.. _{link_target}:' + '\n\n' + section
        body += section
    return body