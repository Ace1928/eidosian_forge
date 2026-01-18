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