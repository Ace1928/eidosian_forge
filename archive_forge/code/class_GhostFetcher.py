import contextlib
from .branch import Branch
from .errors import CommandError, NoSuchRevision
from .trace import note
class GhostFetcher:

    @classmethod
    def from_cmdline(cls, other):
        this_branch = Branch.open_containing('.')[0]
        if other is None:
            other = this_branch.get_parent()
            if other is None:
                raise CommandError('No branch specified and no location saved.')
            else:
                note('Using saved location %s.', other)
        other_branch = Branch.open_containing(other)[0]
        return cls(this_branch, other_branch)

    def __init__(self, this_branch, other_branch):
        self.this_branch = this_branch
        self.other_branch = other_branch

    def run(self):
        lock_other = self.this_branch.base != self.other_branch.base
        with contextlib.ExitStack() as exit_stack:
            exit_stack.enter_context(self.this_branch.lock_write())
            if lock_other:
                exit_stack.enter_context(self.other_branch.lock_read())
            return self._run_locked()

    def iter_ghosts(self):
        """Find all ancestors that aren't stored in this branch."""
        seen = set()
        lines = [self.this_branch.last_revision()]
        if lines[0] is None:
            return
        while len(lines) > 0:
            new_lines = []
            for line in lines:
                if line in seen:
                    continue
                seen.add(line)
                try:
                    revision = self.this_branch.repository.get_revision(line)
                    new_lines.extend(revision.parent_ids)
                except NoSuchRevision:
                    yield line
            lines = new_lines

    def _run_locked(self):
        installed = []
        failed = []
        if self.this_branch.last_revision() is None:
            print('No revisions in branch.')
            return
        ghosts = self.iter_ghosts()
        for revision in ghosts:
            try:
                self.this_branch.fetch(self.other_branch, revision)
                installed.append(revision)
            except NoSuchRevision:
                failed.append(revision)
        return (installed, failed)