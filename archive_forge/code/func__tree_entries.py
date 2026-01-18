import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _tree_entries(path: str, tree: Tree) -> List[TreeEntry]:
    result: List[TreeEntry] = []
    if not tree:
        return result
    for entry in tree.iteritems(name_order=True):
        result.append(entry.in_path(path))
    return result