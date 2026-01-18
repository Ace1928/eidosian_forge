import gzip
import re
from dulwich.refs import SymrefLoop
from .. import config, debug, errors, osutils, trace, ui, urlutils
from ..controldir import BranchReferenceLoop
from ..errors import (AlreadyBranchError, BzrError, ConnectionReset,
from ..push import PushResult
from ..revision import NULL_REVISION
from ..revisiontree import RevisionTree
from ..transport import (NoSuchFile, Transport,
from . import is_github_url, lazy_check_versions, user_agent_for_github
import os
import select
import urllib.parse as urlparse
import dulwich
import dulwich.client
from dulwich.errors import GitProtocolError, HangupException
from dulwich.pack import (PACK_SPOOL_FILE_MAX_SIZE, Pack, load_pack_index,
from dulwich.protocol import ZERO_SHA
from dulwich.refs import SYMREF, DictRefsContainer
from dulwich.repo import NotGitRepository
from .branch import (GitBranch, GitBranchFormat, GitBranchPushResult, GitTags,
from .dir import GitControlDirFormat, GitDir
from .errors import GitSmartRemoteNotSupported
from .mapping import encode_git_path, mapping_registry
from .object_store import get_object_store
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_peeled, ref_to_tag_name,
from .repository import GitRepository, GitRepositoryFormat
class SSHGitSmartTransport(GitSmartTransport):
    _scheme = 'git+ssh'

    def _get_path(self):
        path = self._stripped_path
        if path.startswith('/~/'):
            return path[3:]
        return path

    def _get_client(self):
        if self._client is not None:
            ret = self._client
            self._client = None
            return ret
        location_config = config.LocationConfig(self.base)
        client = dulwich.client.SSHGitClient(self._host, self._port, self._username, report_activity=self._report_activity)
        upload_pack = location_config.get_user_option('git_upload_pack')
        if upload_pack:
            client.alternative_paths['upload-pack'] = upload_pack
        receive_pack = location_config.get_user_option('git_receive_pack')
        if receive_pack:
            client.alternative_paths['receive-pack'] = receive_pack
        return client