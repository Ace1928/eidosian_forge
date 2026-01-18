from .core import encode, decode, alabel, ulabel, IDNAError
import codecs
import re
from typing import Any, Tuple, Optional
def _buffer_decode(self, data: Any, errors: str, final: bool) -> Tuple[str, int]:
    if errors != 'strict':
        raise IDNAError('Unsupported error handling "{}"'.format(errors))
    if not data:
        return ('', 0)
    if not isinstance(data, str):
        data = str(data, 'ascii')
    labels = _unicode_dots_re.split(data)
    trailing_dot = ''
    if labels:
        if not labels[-1]:
            trailing_dot = '.'
            del labels[-1]
        elif not final:
            del labels[-1]
            if labels:
                trailing_dot = '.'
    result = []
    size = 0
    for label in labels:
        result.append(ulabel(label))
        if size:
            size += 1
        size += len(label)
    result_str = '.'.join(result) + trailing_dot
    size += len(trailing_dot)
    return (result_str, size)