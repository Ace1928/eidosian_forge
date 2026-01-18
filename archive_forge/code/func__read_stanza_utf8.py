import re
from typing import Iterator, Optional
from .rio import Stanza
def _read_stanza_utf8(line_iter: Iterator[bytes]) -> Optional[Stanza]:
    stanza = Stanza()
    tag = None
    accum_value = None
    for bline in line_iter:
        if not isinstance(bline, bytes):
            raise TypeError(bline)
        line = bline.decode('utf-8', 'surrogateescape')
        if line is None or line == '':
            break
        if line == '\n':
            break
        real_l = line
        if line[0] == '\t':
            if tag is None:
                raise ValueError('invalid continuation line %r' % real_l)
            accum_value.append('\n' + line[1:-1])
        else:
            if tag is not None:
                stanza.add(tag, ''.join(accum_value))
            try:
                colon_index = line.index(': ')
            except ValueError:
                raise ValueError('tag/value separator not found in line %r' % real_l)
            tag = str(line[:colon_index])
            if not _valid_tag(tag):
                raise ValueError('invalid rio tag {!r}'.format(tag))
            accum_value = [line[colon_index + 2:-1]]
    if tag is not None:
        stanza.add(tag, ''.join(accum_value))
        return stanza
    else:
        return None