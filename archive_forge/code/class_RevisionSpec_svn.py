from ...revisionspec import InvalidRevisionSpec, RevisionSpec
class RevisionSpec_svn(RevisionSpec):
    """Selects a revision using a Subversion revision number."""
    help_txt = 'Selects a revision using a Subversion revision number (revno).\n\n    Subversion revision numbers are per-repository whereas Bazaar revision\n    numbers are per-branch. This revision specifier allows specifying\n    a Subversion revision number.\n    '
    prefix = 'svn:'

    def _match_on(self, branch, revs):
        raise InvalidRevisionSpec(self.user_spec, branch)

    def needs_branch(self):
        return True

    def get_branch(self):
        return None