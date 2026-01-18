import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
class BisectLog:
    """Bisect log file handler."""

    def __init__(self, controldir, filename=BISECT_INFO_PATH):
        self._items = []
        self._current = BisectCurrent(controldir)
        self._controldir = controldir
        self._branch = None
        self._high_revid = None
        self._low_revid = None
        self._middle_revid = None
        self._filename = filename
        self.load()

    def _open_for_read(self):
        """Open log file for reading."""
        if self._filename:
            return self._controldir.control_transport.get(self._filename)
        else:
            return sys.stdin

    def _load_tree(self):
        """Load bzr information."""
        if not self._branch:
            self._branch = self._controldir.open_branch()

    def _find_range_and_middle(self, branch_last_rev=None):
        """Find the current revision range, and the midpoint."""
        self._load_tree()
        self._middle_revid = None
        if not branch_last_rev:
            last_revid = self._branch.last_revision()
        else:
            last_revid = branch_last_rev
        repo = self._branch.repository
        with repo.lock_read():
            graph = repo.get_graph()
            rev_sequence = graph.iter_lefthand_ancestry(last_revid, (_mod_revision.NULL_REVISION,))
            high_revid = None
            low_revid = None
            between_revs = []
            for revision in rev_sequence:
                between_revs.insert(0, revision)
                matches = [x[1] for x in self._items if x[0] == revision and x[1] in ('yes', 'no')]
                if not matches:
                    continue
                if len(matches) > 1:
                    raise RuntimeError('revision %s duplicated' % revision)
                if matches[0] == 'yes':
                    high_revid = revision
                    between_revs = []
                elif matches[0] == 'no':
                    low_revid = revision
                    del between_revs[0]
                    break
            if not high_revid:
                high_revid = last_revid
            if not low_revid:
                low_revid = self._branch.get_rev_id(1)
        spread = len(between_revs) + 1
        if spread < 2:
            middle_index = 0
        else:
            middle_index = spread // 2 - 1
        if len(between_revs) > 0:
            self._middle_revid = between_revs[middle_index]
        else:
            self._middle_revid = high_revid
        self._high_revid = high_revid
        self._low_revid = low_revid

    def _switch_wc_to_revno(self, revno, outf):
        """Move the working tree to the given revno."""
        self._current.switch(revno)
        self._current.show_rev_log(outf=outf)

    def _set_status(self, revid, status):
        """Set the bisect status for the given revid."""
        if not self.is_done():
            if status != 'done' and revid in [x[0] for x in self._items if x[1] in ['yes', 'no']]:
                raise RuntimeError('attempting to add revid %s twice' % revid)
            self._items.append((revid, status))

    def change_file_name(self, filename):
        """Switch log files."""
        self._filename = filename

    def load(self):
        """Load the bisection log."""
        self._items = []
        if self._controldir.control_transport.has(self._filename):
            revlog = self._open_for_read()
            for line in revlog:
                revid, status = line.split()
                self._items.append((revid, status.decode('ascii')))

    def save(self):
        """Save the bisection log."""
        contents = b''.join((b'%s %s\n' % (revid, status.encode('ascii')) for revid, status in self._items))
        if self._filename:
            self._controldir.control_transport.put_bytes(self._filename, contents)
        else:
            sys.stdout.write(contents)

    def is_done(self):
        """Report whether we've found the right revision."""
        return len(self._items) > 0 and self._items[-1][1] == 'done'

    def set_status_from_revspec(self, revspec, status):
        """Set the bisection status for the revision in revspec."""
        self._load_tree()
        revid = revspec[0].in_history(self._branch).rev_id
        self._set_status(revid, status)

    def set_current(self, status):
        """Set the current revision to the given bisection status."""
        self._set_status(self._current.get_current_revid(), status)

    def is_merge_point(self, revid):
        return len(self.get_parent_revids(revid)) > 1

    def get_parent_revids(self, revid):
        repo = self._branch.repository
        with repo.lock_read():
            retval = repo.get_parent_map([revid]).get(revid, None)
        return retval

    def bisect(self, outf):
        """Using the current revision's status, do a bisection."""
        self._find_range_and_middle()
        while (self._middle_revid == self._high_revid or self._middle_revid == self._low_revid) and self.is_merge_point(self._middle_revid):
            for parent in self.get_parent_revids(self._middle_revid):
                if parent == self._low_revid:
                    continue
                else:
                    self._find_range_and_middle(parent)
                    break
        self._switch_wc_to_revno(self._middle_revid, outf)
        if self._middle_revid == self._high_revid or self._middle_revid == self._low_revid:
            self.set_current('done')