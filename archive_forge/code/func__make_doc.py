import inspect
import os
import sys
import warnings
from functools import partial
from importlib.metadata import entry_points
from ..exception import NetworkXNotImplemented
def _make_doc(self):
    if not self.backends:
        return self._orig_doc
    lines = ['Backends', '--------']
    for backend in sorted(self.backends):
        info = backend_info[backend]
        if 'short_summary' in info:
            lines.append(f'{backend} : {info['short_summary']}')
        else:
            lines.append(backend)
        if 'functions' not in info or self.name not in info['functions']:
            lines.append('')
            continue
        func_info = info['functions'][self.name]
        if 'extra_docstring' in func_info:
            lines.extend((f'  {line}' if line else line for line in func_info['extra_docstring'].split('\n')))
            add_gap = True
        else:
            add_gap = False
        if 'extra_parameters' in func_info:
            if add_gap:
                lines.append('')
            lines.append('  Extra parameters:')
            extra_parameters = func_info['extra_parameters']
            for param in sorted(extra_parameters):
                lines.append(f'    {param}')
                if (desc := extra_parameters[param]):
                    lines.append(f'      {desc}')
                lines.append('')
        else:
            lines.append('')
    lines.pop()
    to_add = '\n    '.join(lines)
    return f'{self._orig_doc.rstrip()}\n\n    {to_add}'