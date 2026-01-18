import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _all_eq(seq, key, value):
    for e in seq:
        if key(e) != value:
            return False
    return True