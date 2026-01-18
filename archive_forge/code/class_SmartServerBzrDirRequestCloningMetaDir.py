import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBzrDirRequestCloningMetaDir(SmartServerRequestBzrDir):

    def do_bzrdir_request(self, require_stacking):
        """Get the format that should be used when cloning from this dir.

        New in 1.13.

        :return: on success, a 3-tuple of network names for (control,
            repository, branch) directories, where '' signifies "not present".
            If this BzrDir contains a branch reference then this will fail with
            BranchReference; clients should resolve branch references before
            calling this RPC.
        """
        try:
            branch_ref = self._bzrdir.get_branch_reference()
        except errors.NotBranchError:
            branch_ref = None
        if branch_ref is not None:
            return FailedSmartServerResponse((b'BranchReference',))
        if require_stacking == b'True':
            require_stacking = True
        else:
            require_stacking = False
        control_format = self._bzrdir.cloning_metadir(require_stacking=require_stacking)
        control_name = control_format.network_name()
        if not control_format.fixed_components:
            branch_name = (b'branch', control_format.get_branch_format().network_name())
            repository_name = control_format.repository_format.network_name()
        else:
            branch_name = (b'branch', b'')
            repository_name = b''
        return SuccessfulSmartServerResponse((control_name, repository_name, branch_name))