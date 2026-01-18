import re
from .. import osutils
from ..iterablefile import IterableFile
def _patch_stanza_iter(line_iter):
    map = {b'\\\\': b'\\', b'\\r': b'\r', b'\\\n': b''}

    def mapget(match):
        return map[match.group(0)]
    last_line = None
    for line in line_iter:
        if line.startswith(b'# '):
            line = line[2:]
        elif line.startswith(b'#'):
            line = line[1:]
        else:
            raise ValueError('bad line {!r}'.format(line))
        if last_line is not None and len(line) > 2:
            line = line[2:]
        line = re.sub(b'\r', b'', line)
        line = re.sub(b'\\\\(.|\n)', mapget, line)
        if last_line is None:
            last_line = line
        else:
            last_line += line
        if last_line[-1:] == b'\n':
            yield last_line
            last_line = None
    if last_line is not None:
        yield last_line