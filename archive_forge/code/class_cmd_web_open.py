from io import StringIO
from ... import branch as _mod_branch
from ... import controldir, errors
from ... import forge as _mod_forge
from ... import log as _mod_log
from ... import missing as _mod_missing
from ... import msgeditor, urlutils
from ...commands import Command
from ...i18n import gettext
from ...option import ListOption, Option, RegistryOption
from ...trace import note, warning
class cmd_web_open(Command):
    __doc__ = 'Open a branch page in your web browser.'
    takes_options = [Option('dry-run', 'Do not actually open the browser. Just say the URL we would use.')]
    takes_args = ['location?']

    def _possible_locations(self, location):
        """Yield possible external locations for the branch at 'location'."""
        yield location
        try:
            branch = _mod_branch.Branch.open_containing(location)[0]
        except errors.NotBranchError:
            return
        branch_url = branch.get_public_branch()
        if branch_url is not None:
            yield branch_url
        branch_url = branch.get_push_location()
        if branch_url is not None:
            yield branch_url

    def _get_web_url(self, location):
        for branch_url in self._possible_locations(location):
            try:
                branch = _mod_branch.Branch.open_containing(branch_url)[0]
            except errors.NotBranchError as e:
                mutter('Unable to open branch %s: %s', branch_url, e)
                continue
            try:
                forge = _mod_forge.get_forge(branch)
            except _mod_forge.UnsupportedForge:
                continue
            return forge.get_web_url(branch)
        raise errors.CommandError('Unable to get web URL for %s' % location)

    def run(self, location=None, dry_run=False):
        if location is None:
            location = '.'
        web_url = self._get_web_url(location)
        note(gettext('Opening %s in web browser') % web_url)
        if not dry_run:
            import webbrowser
            webbrowser.open(web_url)