import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _all_same(seq, key):
    return _all_eq(seq[1:], key, key(seq[0]))