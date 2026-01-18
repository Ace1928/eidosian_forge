import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _should_find_content_renames(self):
    return len(self._adds) * len(self._deletes) <= self._max_files ** 2