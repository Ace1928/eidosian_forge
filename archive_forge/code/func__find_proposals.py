from ... import branch as _mod_branch
from ... import controldir, trace
from ...commands import Command
from ...errors import CommandError, NotBranchError
from ...i18n import gettext
from ...option import ListOption, Option
def _find_proposals(self, revision_id, pb):
    from . import lp_api, uris
    lp_base_url = uris.LPNET_SERVICE_ROOT
    launchpad = lp_api.connect_launchpad(lp_base_url, version='devel')
    pb.update(gettext('Finding proposals'))
    return list(launchpad.branches.getMergeProposals(merged_revision=revision_id.decode('utf-8')))