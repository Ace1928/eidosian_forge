import abc
import torch
import itertools
import collections
from torch.nn.modules.module import _addindent
def _hide_packed_params_repr(self, params):
    extra_lines = []
    extra_repr = self.extra_repr()
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    for key, module in self._modules.items():
        if isinstance(module, params):
            continue
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines
    main_str = self._get_name() + '('
    if lines:
        if len(extra_lines) == 1 and (not child_lines):
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'
    main_str += ')'
    return main_str