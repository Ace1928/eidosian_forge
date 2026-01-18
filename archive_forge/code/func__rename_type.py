import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _rename_type(self, check_paths, delete, add):
    if check_paths and delete.old.path == add.new.path:
        return CHANGE_MODIFY
    elif delete.type != CHANGE_DELETE:
        return CHANGE_COPY
    return CHANGE_RENAME