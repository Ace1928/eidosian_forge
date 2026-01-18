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
class PullResult(_Result):
    """Result of a Branch.pull operation.

    Attributes:
      old_revno: Revision number before pull.
      new_revno: Revision number after pull.
      old_revid: Tip revision id before pull.
      new_revid: Tip revision id after pull.
      source_branch: Source (local) branch object. (read locked)
      master_branch: Master branch of the target, or the target if no
        Master
      local_branch: target branch if there is a Master, else None
      target_branch: Target/destination branch object. (write locked)
      tag_conflicts: A list of tag conflicts, see BasicTags.merge_to
      tag_updates: A dict with new tags, see BasicTags.merge_to
    """
    old_revno: Union[int, property]
    new_revno: Union[int, property]
    old_revid: RevisionID
    new_revid: RevisionID
    source_branch: Branch
    master_branch: Branch
    local_branch: Optional[Branch]
    target_branch: Branch
    tag_conflicts: List['TagConflict']
    tag_updates: 'TagUpdates'

    def report(self, to_file: TextIO) -> None:
        tag_conflicts = getattr(self, 'tag_conflicts', None)
        tag_updates = getattr(self, 'tag_updates', None)
        if not is_quiet():
            if self.old_revid != self.new_revid:
                to_file.write('Now on revision %d.\n' % self.new_revno)
            if tag_updates:
                to_file.write('%d tag(s) updated.\n' % len(tag_updates))
            if self.old_revid == self.new_revid and (not tag_updates):
                if not tag_conflicts:
                    to_file.write('No revisions or tags to pull.\n')
                else:
                    to_file.write('No revisions to pull.\n')
        self._show_tag_conficts(to_file)