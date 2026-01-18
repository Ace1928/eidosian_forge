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
def _check_already_merged(branch, target):
    if branch.last_revision() == target.last_revision():
        raise errors.CommandError(gettext('All local changes are already present in target.'))