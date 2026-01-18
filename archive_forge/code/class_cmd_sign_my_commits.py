from . import controldir, errors, gpg
from . import repository as _mod_repository
from . import revision as _mod_revision
from .commands import Command
from .i18n import gettext, ngettext
from .option import Option
class cmd_sign_my_commits(Command):
    __doc__ = 'Sign all commits by a given committer.\n\n    If location is not specified the local tree is used.\n    If committer is not specified the default committer is used.\n\n    This does not sign commits that already have signatures.\n    '
    takes_options = [Option('dry-run', help="Don't actually sign anything, just print the revisions that would be signed.")]
    takes_args = ['location?', 'committer?']

    def run(self, location=None, committer=None, dry_run=False):
        if location is None:
            bzrdir = controldir.ControlDir.open_containing('.')[0]
        else:
            bzrdir = controldir.ControlDir.open(location)
        branch = bzrdir.open_branch()
        repo = branch.repository
        branch_config = branch.get_config_stack()
        if committer is None:
            committer = branch_config.get('email')
        gpg_strategy = gpg.GPGStrategy(branch_config)
        count = 0
        with repo.lock_write():
            graph = repo.get_graph()
            with _mod_repository.WriteGroup(repo):
                for rev_id, parents in graph.iter_ancestry([branch.last_revision()]):
                    if _mod_revision.is_null(rev_id):
                        continue
                    if parents is None:
                        continue
                    if repo.has_signature_for_revision_id(rev_id):
                        continue
                    rev = repo.get_revision(rev_id)
                    if rev.committer != committer:
                        continue
                    self.outf.write('%s\n' % rev_id)
                    count += 1
                    if not dry_run:
                        repo.sign_revision(rev_id, gpg_strategy)
        self.outf.write(ngettext('Signed %d revision.\n', 'Signed %d revisions.\n', count) % count)