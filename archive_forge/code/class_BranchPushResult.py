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
class BranchPushResult(_Result):
    """Result of a Branch.push operation.

    Attributes:
      old_revno: Revision number (eg 10) of the target before push.
      new_revno: Revision number (eg 12) of the target after push.
      old_revid: Tip revision id (eg joe@foo.com-1234234-aoeua34) of target
        before the push.
      new_revid: Tip revision id (eg joe@foo.com-5676566-boa234a) of target
        after the push.
      source_branch: Source branch object that the push was from. This is
        read locked, and generally is a local (and thus low latency) branch.
      master_branch: If target is a bound branch, the master branch of
        target, or target itself. Always write locked.
      target_branch: The direct Branch where data is being sent (write
        locked).
      local_branch: If the target is a bound branch this will be the
        target, otherwise it will be None.
    """
    old_revno: int
    new_revno: int
    old_revid: RevisionID
    new_revid: RevisionID
    source_branch: Branch
    master_branch: Branch
    target_branch: Branch
    local_branch: Optional[Branch]

    def report(self, to_file: TextIO) -> None:
        from breezy.i18n import gettext, ngettext
        tag_conflicts = getattr(self, 'tag_conflicts', None)
        tag_updates = getattr(self, 'tag_updates', None)
        if not is_quiet():
            if self.old_revid != self.new_revid:
                if self.new_revno is not None:
                    note(gettext('Pushed up to revision %d.'), self.new_revno)
                else:
                    note(gettext('Pushed up to revision id %s.'), self.new_revid.decode('utf-8'))
            if tag_updates:
                note(ngettext('%d tag updated.', '%d tags updated.', len(tag_updates)) % len(tag_updates))
            if self.old_revid == self.new_revid and (not tag_updates):
                if not tag_conflicts:
                    note(gettext('No new revisions or tags to push.'))
                else:
                    note(gettext('No new revisions to push.'))
        self._show_tag_conficts(to_file)