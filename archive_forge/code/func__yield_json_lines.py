from typing import Union, Iterable, Sequence, Any, Optional, Iterator
import sys
import json as _builtin_json
import gzip
from . import ujson
from .util import force_path, force_string, FilePath, JSONInput, JSONOutput
def _yield_json_lines(stream: Iterable[str], skip: bool=False) -> Iterable[JSONOutput]:
    line_no = 1
    for line in stream:
        line = line.strip()
        if line == '':
            continue
        try:
            yield ujson.loads(line)
        except ValueError:
            if skip:
                continue
            raise ValueError(f'Invalid JSON on line {line_no}: {line}')
        line_no += 1