import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
class BisectCurrent:
    """Bisect class for managing the current revision."""

    def __init__(self, controldir, filename=BISECT_REV_PATH):
        self._filename = filename
        self._controldir = controldir
        self._branch = self._controldir.open_branch()
        if self._controldir.control_transport.has(filename):
            self._revid = self._controldir.control_transport.get_bytes(filename).strip()
        else:
            self._revid = self._branch.last_revision()

    def _save(self):
        """Save the current revision."""
        self._controldir.control_transport.put_bytes(self._filename, self._revid + b'\n')

    def get_current_revid(self):
        """Return the current revision id."""
        return self._revid

    def get_current_revno(self):
        """Return the current revision number as a tuple."""
        return self._branch.revision_id_to_dotted_revno(self._revid)

    def get_parent_revids(self):
        """Return the IDs of the current revision's predecessors."""
        repo = self._branch.repository
        with repo.lock_read():
            retval = repo.get_parent_map([self._revid]).get(self._revid, None)
        return retval

    def is_merge_point(self):
        """Is the current revision a merge point?"""
        return len(self.get_parent_revids()) > 1

    def show_rev_log(self, outf):
        """Write the current revision's log entry to a file."""
        rev = self._branch.repository.get_revision(self._revid)
        revno = '.'.join([str(x) for x in self.get_current_revno()])
        outf.write('On revision {} ({}):\n{}\n'.format(revno, rev.revision_id, rev.message))

    def switch(self, revid):
        """Switch the current revision to the given revid."""
        working = self._controldir.open_workingtree()
        if isinstance(revid, int):
            revid = self._branch.get_rev_id(revid)
        elif isinstance(revid, list):
            revid = revid[0].in_history(working.branch).rev_id
        working.revert(None, working.branch.repository.revision_tree(revid), False)
        self._revid = revid
        self._save()

    def reset(self):
        """Revert bisection, setting the working tree to normal."""
        working = self._controldir.open_workingtree()
        last_rev = working.branch.last_revision()
        rev_tree = working.branch.repository.revision_tree(last_rev)
        working.revert(None, rev_tree, False)
        if self._controldir.control_transport.has(BISECT_REV_PATH):
            self._controldir.control_transport.delete(BISECT_REV_PATH)