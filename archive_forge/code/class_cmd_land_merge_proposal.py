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
class cmd_land_merge_proposal(Command):
    __doc__ = 'Land a merge proposal.'
    takes_args = ['url']
    takes_options = [Option('message', help='Commit message to use.', type=str)]

    def run(self, url, message=None):
        proposal = _mod_forge.get_proposal_by_url(url)
        proposal.merge(commit_message=message)