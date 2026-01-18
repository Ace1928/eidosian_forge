import bz2
import os
import re
import sys
import zlib
from typing import Callable, List, Optional
import fastbencode as bencode
from .. import branch
from .. import bzr as _mod_bzr
from .. import config as _mod_config
from .. import (controldir, debug, errors, gpg, graph, lock, lockdir, osutils,
from .. import repository as _mod_repository
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..branch import BranchWriteLockResult
from ..decorators import only_raises
from ..errors import NoSuchRevision, SmartProtocolError
from ..i18n import gettext
from ..lockable_files import LockableFiles
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..revision import NULL_REVISION
from ..trace import log_exception_quietly, mutter, note, warning
from . import branch as bzrbranch
from . import bzrdir as _mod_bzrdir
from . import inventory_delta
from . import repository as bzrrepository
from . import testament as _mod_testament
from . import vf_repository, vf_search
from .branch import BranchReferenceFormat
from .inventory import Inventory
from .inventorytree import InventoryRevisionTree
from .serializer import format_registry as serializer_format_registry
from .smart import client
from .smart import repository as smart_repo
from .smart import vfs
from .smart.client import _SmartClient
from .versionedfile import FulltextContentFactory
class RemoteBzrDir(_mod_bzrdir.BzrDir, _RpcHelper):
    """Control directory on a remote server, accessed via bzr:// or similar."""

    @property
    def user_transport(self):
        return self.root_transport

    @property
    def control_transport(self):
        return self.transport

    def __init__(self, transport, format, _client=None, _force_probe=False):
        """Construct a RemoteBzrDir.

        :param _client: Private parameter for testing. Disables probing and the
            use of a real bzrdir.
        """
        _mod_bzrdir.BzrDir.__init__(self, transport, format)
        self._real_bzrdir = None
        self._has_working_tree = None
        self._next_open_branch_result = None
        if _client is None:
            medium = transport.get_smart_medium()
            self._client = client._SmartClient(medium)
        else:
            self._client = _client
            if not _force_probe:
                return
        self._probe_bzrdir()

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self._client)

    def _probe_bzrdir(self):
        medium = self._client._medium
        path = self._path_for_remote_call(self._client)
        if medium._is_remote_before((2, 1)):
            self._rpc_open(path)
            return
        try:
            self._rpc_open_2_1(path)
            return
        except errors.UnknownSmartMethod:
            medium._remember_remote_is_before((2, 1))
            self._rpc_open(path)

    def _rpc_open_2_1(self, path):
        response = self._call(b'BzrDir.open_2.1', path)
        if response == (b'no',):
            raise errors.NotBranchError(path=self.root_transport.base)
        elif response[0] == b'yes':
            if response[1] == b'yes':
                self._has_working_tree = True
            elif response[1] == b'no':
                self._has_working_tree = False
            else:
                raise errors.UnexpectedSmartServerResponse(response)
        else:
            raise errors.UnexpectedSmartServerResponse(response)

    def _rpc_open(self, path):
        response = self._call(b'BzrDir.open', path)
        if response not in [(b'yes',), (b'no',)]:
            raise errors.UnexpectedSmartServerResponse(response)
        if response == (b'no',):
            raise errors.NotBranchError(path=self.root_transport.base)

    def _ensure_real(self):
        """Ensure that there is a _real_bzrdir set.

        Used before calls to self._real_bzrdir.
        """
        if not self._real_bzrdir:
            if 'hpssvfs' in debug.debug_flags:
                import traceback
                warning('VFS BzrDir access triggered\n%s', ''.join(traceback.format_stack()))
            self._real_bzrdir = _mod_bzrdir.BzrDir.open_from_transport(self.root_transport, probers=[_mod_bzr.BzrProber])
            self._format._network_name = self._real_bzrdir._format.network_name()

    def _translate_error(self, err, **context):
        _translate_error(err, bzrdir=self, **context)

    def break_lock(self):
        self._next_open_branch_result = None
        return _mod_bzrdir.BzrDir.break_lock(self)

    def _vfs_checkout_metadir(self):
        self._ensure_real()
        return self._real_bzrdir.checkout_metadir()

    def checkout_metadir(self):
        """Retrieve the controldir format to use for checkouts of this one.
        """
        medium = self._client._medium
        if medium._is_remote_before((2, 5)):
            return self._vfs_checkout_metadir()
        path = self._path_for_remote_call(self._client)
        try:
            response = self._client.call(b'BzrDir.checkout_metadir', path)
        except errors.UnknownSmartMethod:
            medium._remember_remote_is_before((2, 5))
            return self._vfs_checkout_metadir()
        if len(response) != 3:
            raise errors.UnexpectedSmartServerResponse(response)
        control_name, repo_name, branch_name = response
        try:
            format = controldir.network_format_registry.get(control_name)
        except KeyError:
            raise errors.UnknownFormatError(kind='control', format=control_name)
        if repo_name:
            try:
                repo_format = _mod_repository.network_format_registry.get(repo_name)
            except KeyError:
                raise errors.UnknownFormatError(kind='repository', format=repo_name)
            format.repository_format = repo_format
        if branch_name:
            try:
                format.set_branch_format(branch.network_format_registry.get(branch_name))
            except KeyError:
                raise errors.UnknownFormatError(kind='branch', format=branch_name)
        return format

    def _vfs_cloning_metadir(self, require_stacking=False):
        self._ensure_real()
        return self._real_bzrdir.cloning_metadir(require_stacking=require_stacking)

    def cloning_metadir(self, require_stacking=False):
        medium = self._client._medium
        if medium._is_remote_before((1, 13)):
            return self._vfs_cloning_metadir(require_stacking=require_stacking)
        verb = b'BzrDir.cloning_metadir'
        if require_stacking:
            stacking = b'True'
        else:
            stacking = b'False'
        path = self._path_for_remote_call(self._client)
        try:
            response = self._call(verb, path, stacking)
        except errors.UnknownSmartMethod:
            medium._remember_remote_is_before((1, 13))
            return self._vfs_cloning_metadir(require_stacking=require_stacking)
        except UnknownErrorFromSmartServer as err:
            if err.error_tuple != (b'BranchReference',):
                raise
            referenced_branch = self.open_branch()
            return referenced_branch.controldir.cloning_metadir()
        if len(response) != 3:
            raise errors.UnexpectedSmartServerResponse(response)
        control_name, repo_name, branch_info = response
        if len(branch_info) != 2:
            raise errors.UnexpectedSmartServerResponse(response)
        branch_ref, branch_name = branch_info
        try:
            format = controldir.network_format_registry.get(control_name)
        except KeyError:
            raise errors.UnknownFormatError(kind='control', format=control_name)
        if repo_name:
            try:
                format.repository_format = _mod_repository.network_format_registry.get(repo_name)
            except KeyError:
                raise errors.UnknownFormatError(kind='repository', format=repo_name)
        if branch_ref == b'ref':
            ref_bzrdir = _mod_bzrdir.BzrDir.open(branch_name)
            branch_format = ref_bzrdir.cloning_metadir().get_branch_format()
            format.set_branch_format(branch_format)
        elif branch_ref == b'branch':
            if branch_name:
                try:
                    branch_format = branch.network_format_registry.get(branch_name)
                except KeyError:
                    raise errors.UnknownFormatError(kind='branch', format=branch_name)
                format.set_branch_format(branch_format)
        else:
            raise errors.UnexpectedSmartServerResponse(response)
        return format

    def create_repository(self, shared=False):
        result = self._format.repository_format.initialize(self, shared)
        if not isinstance(result, RemoteRepository):
            return self.open_repository()
        else:
            return result

    def destroy_repository(self):
        """See BzrDir.destroy_repository"""
        path = self._path_for_remote_call(self._client)
        try:
            response = self._call(b'BzrDir.destroy_repository', path)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            self._real_bzrdir.destroy_repository()
            return
        if response[0] != b'ok':
            raise SmartProtocolError('unexpected response code {}'.format(response))

    def create_branch(self, name=None, repository=None, append_revisions_only=None):
        if name is None:
            name = self._get_selected_branch()
        if name != '':
            raise controldir.NoColocatedBranchSupport(self)
        real_branch = self._format.get_branch_format().initialize(self, name=name, repository=repository, append_revisions_only=append_revisions_only)
        if not isinstance(real_branch, RemoteBranch):
            if not isinstance(repository, RemoteRepository):
                raise AssertionError('need a RemoteRepository to use with RemoteBranch, got %r' % (repository,))
            result = RemoteBranch(self, repository, real_branch, name=name)
        else:
            result = real_branch
        self._next_open_branch_result = result
        return result

    def destroy_branch(self, name=None):
        """See BzrDir.destroy_branch"""
        if name is None:
            name = self._get_selected_branch()
        if name != '':
            raise controldir.NoColocatedBranchSupport(self)
        path = self._path_for_remote_call(self._client)
        try:
            if name != '':
                args = (name,)
            else:
                args = ()
            response = self._call(b'BzrDir.destroy_branch', path, *args)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            self._real_bzrdir.destroy_branch(name=name)
            self._next_open_branch_result = None
            return
        self._next_open_branch_result = None
        if response[0] != b'ok':
            raise SmartProtocolError('unexpected response code {}'.format(response))

    def create_workingtree(self, revision_id=None, from_branch=None, accelerator_tree=None, hardlink=False):
        raise errors.NotLocalUrl(self.transport.base)

    def find_branch_format(self, name=None):
        """Find the branch 'format' for this bzrdir.

        This might be a synthetic object for e.g. RemoteBranch and SVN.
        """
        b = self.open_branch(name=name)
        return b._format

    def branch_names(self):
        path = self._path_for_remote_call(self._client)
        try:
            response, handler = self._call_expecting_body(b'BzrDir.get_branches', path)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_bzrdir.branch_names()
        if response[0] != b'success':
            raise errors.UnexpectedSmartServerResponse(response)
        body = bencode.bdecode(handler.read_body_bytes())
        ret = []
        for name, value in body.items():
            name = name.decode('utf-8')
            ret.append(name)
        return ret

    def get_branches(self, possible_transports=None, ignore_fallbacks=False):
        path = self._path_for_remote_call(self._client)
        try:
            response, handler = self._call_expecting_body(b'BzrDir.get_branches', path)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_bzrdir.get_branches()
        if response[0] != b'success':
            raise errors.UnexpectedSmartServerResponse(response)
        body = bencode.bdecode(handler.read_body_bytes())
        ret = {}
        for name, value in body.items():
            name = name.decode('utf-8')
            ret[name] = self._open_branch(name, value[0].decode('ascii'), value[1], possible_transports=possible_transports, ignore_fallbacks=ignore_fallbacks)
        return ret

    def set_branch_reference(self, target_branch, name=None):
        """See BzrDir.set_branch_reference()."""
        if name is None:
            name = self._get_selected_branch()
        if name != '':
            raise controldir.NoColocatedBranchSupport(self)
        self._ensure_real()
        return self._real_bzrdir.set_branch_reference(target_branch, name=name)

    def get_branch_reference(self, name=None):
        """See BzrDir.get_branch_reference()."""
        if name is None:
            name = self._get_selected_branch()
        if name != '':
            raise controldir.NoColocatedBranchSupport(self)
        response = self._get_branch_reference()
        if response[0] == 'ref':
            return response[1].decode('utf-8')
        else:
            return None

    def _get_branch_reference(self):
        """Get branch reference information

        :return: Tuple with (kind, location_or_format)
            if kind == 'ref', then location_or_format contains a location
            otherwise, it contains a format name
        """
        path = self._path_for_remote_call(self._client)
        medium = self._client._medium
        candidate_calls = [(b'BzrDir.open_branchV3', (2, 1)), (b'BzrDir.open_branchV2', (1, 13)), (b'BzrDir.open_branch', None)]
        for verb, required_version in candidate_calls:
            if required_version and medium._is_remote_before(required_version):
                continue
            try:
                response = self._call(verb, path)
            except errors.UnknownSmartMethod:
                if required_version is None:
                    raise
                medium._remember_remote_is_before(required_version)
            else:
                break
        if verb == b'BzrDir.open_branch':
            if response[0] != b'ok':
                raise errors.UnexpectedSmartServerResponse(response)
            if response[1] != b'':
                return ('ref', response[1])
            else:
                return ('branch', b'')
        if response[0] not in (b'ref', b'branch'):
            raise errors.UnexpectedSmartServerResponse(response)
        return (response[0].decode('ascii'), response[1])

    def _get_tree_branch(self, name=None):
        """See BzrDir._get_tree_branch()."""
        return (None, self.open_branch(name=name))

    def _open_branch(self, name, kind, location_or_format, ignore_fallbacks=False, possible_transports=None):
        if kind == 'ref':
            format = BranchReferenceFormat()
            ref_loc = urlutils.join(self.user_url, location_or_format.decode('utf-8'))
            return format.open(self, name=name, _found=True, location=ref_loc, ignore_fallbacks=ignore_fallbacks, possible_transports=possible_transports)
        branch_format_name = location_or_format
        if not branch_format_name:
            branch_format_name = None
        format = RemoteBranchFormat(network_name=branch_format_name)
        return RemoteBranch(self, self.find_repository(), format=format, setup_stacking=not ignore_fallbacks, name=name, possible_transports=possible_transports)

    def open_branch(self, name=None, unsupported=False, ignore_fallbacks=False, possible_transports=None):
        if name is None:
            name = self._get_selected_branch()
        if name != '':
            raise controldir.NoColocatedBranchSupport(self)
        if unsupported:
            raise NotImplementedError('unsupported flag support not implemented yet.')
        if self._next_open_branch_result is not None:
            result = self._next_open_branch_result
            self._next_open_branch_result = None
            return result
        response = self._get_branch_reference()
        return self._open_branch(name, response[0], response[1], possible_transports=possible_transports, ignore_fallbacks=ignore_fallbacks)

    def _open_repo_v1(self, path):
        verb = b'BzrDir.find_repository'
        response = self._call(verb, path)
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        self._ensure_real()
        repo = self._real_bzrdir.open_repository()
        response = response + (b'no', repo._format.network_name())
        return (response, repo)

    def _open_repo_v2(self, path):
        verb = b'BzrDir.find_repositoryV2'
        response = self._call(verb, path)
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        self._ensure_real()
        repo = self._real_bzrdir.open_repository()
        response = response + (repo._format.network_name(),)
        return (response, repo)

    def _open_repo_v3(self, path):
        verb = b'BzrDir.find_repositoryV3'
        medium = self._client._medium
        if medium._is_remote_before((1, 13)):
            raise errors.UnknownSmartMethod(verb)
        try:
            response = self._call(verb, path)
        except errors.UnknownSmartMethod:
            medium._remember_remote_is_before((1, 13))
            raise
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        return (response, None)

    def open_repository(self):
        path = self._path_for_remote_call(self._client)
        response = None
        for probe in [self._open_repo_v3, self._open_repo_v2, self._open_repo_v1]:
            try:
                response, real_repo = probe(path)
                break
            except errors.UnknownSmartMethod:
                pass
        if response is None:
            raise errors.UnknownSmartMethod(b'BzrDir.find_repository{3,2,}')
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        if len(response) != 6:
            raise SmartProtocolError('incorrect response length {}'.format(response))
        if response[1] == b'':
            format = response_tuple_to_repo_format(response[2:])
            format._creating_bzrdir = self
            remote_repo = RemoteRepository(self, format)
            format._creating_repo = remote_repo
            if real_repo is not None:
                remote_repo._set_real_repository(real_repo)
            return remote_repo
        else:
            raise errors.NoRepositoryPresent(self)

    def has_workingtree(self):
        if self._has_working_tree is None:
            path = self._path_for_remote_call(self._client)
            try:
                response = self._call(b'BzrDir.has_workingtree', path)
            except errors.UnknownSmartMethod:
                self._ensure_real()
                self._has_working_tree = self._real_bzrdir.has_workingtree()
            else:
                if response[0] not in (b'yes', b'no'):
                    raise SmartProtocolError('unexpected response code {}'.format(response))
                self._has_working_tree = response[0] == b'yes'
        return self._has_working_tree

    def open_workingtree(self, recommend_upgrade=True):
        if self.has_workingtree():
            raise errors.NotLocalUrl(self.root_transport)
        else:
            raise errors.NoWorkingTree(self.root_transport.base)

    def _path_for_remote_call(self, client):
        """Return the path to be used for this bzrdir in a remote call."""
        remote_path = client.remote_path_from_transport(self.root_transport)
        remote_path = remote_path.decode('utf-8')
        base_url, segment_parameters = urlutils.split_segment_parameters_raw(remote_path)
        base_url = base_url.encode('utf-8')
        return base_url

    def get_branch_transport(self, branch_format, name=None):
        self._ensure_real()
        return self._real_bzrdir.get_branch_transport(branch_format, name=name)

    def get_repository_transport(self, repository_format):
        self._ensure_real()
        return self._real_bzrdir.get_repository_transport(repository_format)

    def get_workingtree_transport(self, workingtree_format):
        self._ensure_real()
        return self._real_bzrdir.get_workingtree_transport(workingtree_format)

    def can_convert_format(self):
        """Upgrading of remote bzrdirs is not supported yet."""
        return False

    def needs_format_conversion(self, format):
        """Upgrading of remote bzrdirs is not supported yet."""
        return False

    def _get_config(self):
        return RemoteBzrDirConfig(self)

    def _get_config_store(self):
        return RemoteControlStore(self)