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