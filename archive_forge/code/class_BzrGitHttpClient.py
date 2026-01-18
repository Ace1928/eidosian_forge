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
class BzrGitHttpClient(dulwich.client.HttpGitClient):

    def __init__(self, transport, *args, **kwargs):
        self.transport = transport
        url = urlutils.URL.from_string(transport.external_url())
        url.user = url.quoted_user = None
        url.password = url.quoted_password = None
        url = urlutils.strip_segment_parameters(str(url))
        super().__init__(url, *args, **kwargs)

    def archive(self, path, committish, write_data, progress=None, write_error=None, format=None, subdirs=None, prefix=None):
        raise GitSmartRemoteNotSupported(self.archive, self)

    def _http_request(self, url, headers=None, data=None, allow_compression=False):
        """Perform HTTP request.

        :param url: Request URL.
        :param headers: Optional custom headers to override defaults.
        :param data: Request data.
        :return: Tuple (`response`, `read`), where response is an `urllib3`
            response object with additional `content_type` and
            `redirect_location` properties, and `read` is a consumable read
            method for the response data.
        """
        if is_github_url(url):
            headers['User-agent'] = user_agent_for_github()
        headers['Pragma'] = 'no-cache'
        response = self.transport.request('GET' if data is None else 'POST', url, body=data, headers=headers, retries=8)
        if response.status == 404:
            raise NotGitRepository()
        elif response.status != 200:
            raise GitProtocolError('unexpected http resp %d for %s' % (response.status, url))
        read = response.read

        class WrapResponse:

            def __init__(self, response):
                self._response = response
                self.status = response.status
                self.content_type = response.getheader('Content-Type')
                self.redirect_location = response._actual.geturl()

            def readlines(self):
                return self._response.readlines()

            def close(self):
                pass
        return (WrapResponse(response), read)