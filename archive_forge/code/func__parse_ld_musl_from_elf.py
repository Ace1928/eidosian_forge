import contextlib
import functools
import operator
import os
import re
import struct
import subprocess
import sys
from typing import IO, Iterator, NamedTuple, Optional, Tuple
def _parse_ld_musl_from_elf(f: IO[bytes]) -> Optional[str]:
    """Detect musl libc location by parsing the Python executable.

    Based on: https://gist.github.com/lyssdod/f51579ae8d93c8657a5564aefc2ffbca
    ELF header: https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.eheader.html
    """
    f.seek(0)
    try:
        ident = _read_unpacked(f, '16B')
    except struct.error:
        return None
    if ident[:4] != tuple(b'\x7fELF'):
        return None
    f.seek(struct.calcsize('HHI'), 1)
    try:
        e_fmt, p_fmt, p_idx = {1: ('IIIIHHH', 'IIIIIIII', (0, 1, 4)), 2: ('QQQIHHH', 'IIQQQQQQ', (0, 2, 5))}[ident[4]]
    except KeyError:
        return None
    else:
        p_get = operator.itemgetter(*p_idx)
    try:
        _, e_phoff, _, _, _, e_phentsize, e_phnum = _read_unpacked(f, e_fmt)
    except struct.error:
        return None
    for i in range(e_phnum + 1):
        f.seek(e_phoff + e_phentsize * i)
        try:
            p_type, p_offset, p_filesz = p_get(_read_unpacked(f, p_fmt))
        except struct.error:
            return None
        if p_type != 3:
            continue
        f.seek(p_offset)
        interpreter = os.fsdecode(f.read(p_filesz)).strip('\x00')
        if 'musl' not in interpreter:
            return None
        return interpreter
    return None