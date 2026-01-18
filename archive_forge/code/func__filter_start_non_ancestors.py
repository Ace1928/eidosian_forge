from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
def _filter_start_non_ancestors(self, rev_iter):
    try:
        first = next(rev_iter)
    except StopIteration:
        return
    rev_id, merge_depth, revno, end_of_merge = first
    yield first
    if not merge_depth:
        yield from rev_iter
    clean = False
    whitelist = set()
    pmap = self.repository.get_parent_map([rev_id])
    parents = pmap.get(rev_id, [])
    if parents:
        whitelist.update(parents)
    else:
        return
    for rev_id, merge_depth, revno, end_of_merge in rev_iter:
        if not clean:
            if rev_id in whitelist:
                pmap = self.repository.get_parent_map([rev_id])
                parents = pmap.get(rev_id, [])
                whitelist.remove(rev_id)
                whitelist.update(parents)
                if merge_depth == 0:
                    clean = True
            else:
                continue
        yield (rev_id, merge_depth, revno, end_of_merge)