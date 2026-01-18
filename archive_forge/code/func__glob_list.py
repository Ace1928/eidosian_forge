import re
from typing import Iterable, Union
@staticmethod
def _glob_list(elems: GlobPattern, separator: str='.'):
    if isinstance(elems, str):
        return [GlobGroup._glob_to_re(elems, separator)]
    else:
        return [GlobGroup._glob_to_re(e, separator) for e in elems]