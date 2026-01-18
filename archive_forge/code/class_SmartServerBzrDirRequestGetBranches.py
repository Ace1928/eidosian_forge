import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBzrDirRequestGetBranches(SmartServerRequestBzrDir):

    def do_bzrdir_request(self):
        """Get the branches in a control directory.

        The body is a bencoded dictionary, with values similar to the return
        value of the open branch request.
        """
        branch_names = self._bzrdir.branch_names()
        ret = {}
        for name in branch_names:
            if name is None:
                name = b''
            branch_ref = self._bzrdir.get_branch_reference(name=name)
            if branch_ref is not None:
                branch_ref = urlutils.relative_url(self._bzrdir.user_url, branch_ref)
                value = (b'ref', branch_ref.encode('utf-8'))
            else:
                b = self._bzrdir.open_branch(name=name, ignore_fallbacks=True)
                value = (b'branch', b._format.network_name())
            ret[name.encode('utf-8')] = value
        return SuccessfulSmartServerResponse((b'success',), bencode.bencode(ret))