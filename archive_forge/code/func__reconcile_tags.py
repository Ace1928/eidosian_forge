import contextlib
import itertools
import re
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple
from . import branch as _mod_branch
from . import errors
from .inter import InterObject
from .registry import Registry
from .revision import RevisionID
def _reconcile_tags(source_dict: Dict[str, bytes], dest_dict: Dict[str, bytes], overwrite: bool, selector: Optional[TagSelector]) -> Tuple[Dict[str, RevisionID], TagUpdates, List[TagConflict]]:
    """Do a two-way merge of two tag dictionaries.

    * only in source => source value
    * only in destination => destination value
    * same definitions => that
    * different definitions => if overwrite is False, keep destination
      value and add to conflict list, otherwise use the source value

    :returns: (result_dict, updates,
        [(conflicting_tag, source_target, dest_target)])
    """
    conflicts = []
    updates = {}
    result = dict(dest_dict)
    for name, target in source_dict.items():
        if selector and (not selector(name)):
            continue
        if result.get(name) == target:
            pass
        elif name not in result or overwrite:
            updates[name] = target
            result[name] = target
        else:
            conflicts.append((name, target, result[name]))
    return (result, updates, conflicts)