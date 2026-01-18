import re
import string
import sys
from jsbeautifier.unpackers import UnpackingError
def _filterargs(source):
    """Juice from a source file the four args needed by decoder."""
    juicers = ["}\\('(.*)', *(\\d+|\\[\\]), *(\\d+), *'(.*)'\\.split\\('\\|'\\), *(\\d+), *(.*)\\)\\)", "}\\('(.*)', *(\\d+|\\[\\]), *(\\d+), *'(.*)'\\.split\\('\\|'\\)"]
    for juicer in juicers:
        args = re.search(juicer, source, re.DOTALL)
        if args:
            a = args.groups()
            if a[1] == '[]':
                a = list(a)
                a[1] = 62
                a = tuple(a)
            try:
                return (a[0], a[3].split('|'), int(a[1]), int(a[2]))
            except ValueError:
                raise UnpackingError('Corrupted p.a.c.k.e.r. data.')
    raise UnpackingError('Could not make sense of p.a.c.k.e.r data (unexpected code structure)')