import re
import textwrap
from ._array_like import NDArray
def _parse_docstrings() -> str:
    """Convert all docstrings in ``_docstrings_list`` into a single
    sphinx-legible text block.

    """
    type_list_ret = []
    for name, value, doc in _docstrings_list:
        s = textwrap.dedent(doc).replace('\n', '\n    ')
        lines = s.split('\n')
        new_lines = []
        indent = ''
        for line in lines:
            m = re.match('^(\\s+)[-=]+\\s*$', line)
            if m and new_lines:
                prev = textwrap.dedent(new_lines.pop())
                if prev == 'Examples':
                    indent = ''
                    new_lines.append(f'{m.group(1)}.. rubric:: {prev}')
                else:
                    indent = 4 * ' '
                    new_lines.append(f'{m.group(1)}.. admonition:: {prev}')
                new_lines.append('')
            else:
                new_lines.append(f'{indent}{line}')
        s = '\n'.join(new_lines)
        s_block = f'.. data:: {name}\n    :value: {value}\n    {s}'
        type_list_ret.append(s_block)
    return '\n'.join(type_list_ret)