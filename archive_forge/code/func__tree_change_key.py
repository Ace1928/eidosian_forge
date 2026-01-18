import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _tree_change_key(entry):
    path1 = entry.old.path
    path2 = entry.new.path
    if path1 is None:
        path1 = path2
    if path2 is None:
        path2 = path1
    return (path1, path2)