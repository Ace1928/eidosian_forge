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
def _initialize_on_transport_ex_rpc(self, client, path, transport, use_existing_dir, create_prefix, force_new_repo, stacked_on, stack_on_pwd, repo_format_name, make_working_trees, shared_repo):
    args = []
    args.append(self._serialize_NoneTrueFalse(use_existing_dir))
    args.append(self._serialize_NoneTrueFalse(create_prefix))
    args.append(self._serialize_NoneTrueFalse(force_new_repo))
    args.append(self._serialize_NoneString(stacked_on))
    if stack_on_pwd:
        try:
            stack_on_pwd = transport.relpath(stack_on_pwd).encode('utf-8')
            if not stack_on_pwd:
                stack_on_pwd = b'.'
        except errors.PathNotChild:
            pass
    args.append(self._serialize_NoneString(stack_on_pwd))
    args.append(self._serialize_NoneString(repo_format_name))
    args.append(self._serialize_NoneTrueFalse(make_working_trees))
    args.append(self._serialize_NoneTrueFalse(shared_repo))
    request_network_name = self._network_name or _mod_bzrdir.BzrDirFormat.get_default_format().network_name()
    try:
        response = client.call(b'BzrDirFormat.initialize_ex_1.16', request_network_name, path, *args)
    except errors.UnknownSmartMethod:
        client._medium._remember_remote_is_before((1, 16))
        local_dir_format = _mod_bzrdir.BzrDirMetaFormat1()
        self._supply_sub_formats_to(local_dir_format)
        return local_dir_format.initialize_on_transport_ex(transport, use_existing_dir=use_existing_dir, create_prefix=create_prefix, force_new_repo=force_new_repo, stacked_on=stacked_on, stack_on_pwd=stack_on_pwd, repo_format_name=repo_format_name, make_working_trees=make_working_trees, shared_repo=shared_repo, vfs_only=True)
    except errors.ErrorFromSmartServer as err:
        _translate_error(err, path=path.decode('utf-8'))
    repo_path = response[0]
    bzrdir_name = response[6]
    require_stacking = response[7]
    require_stacking = self.parse_NoneTrueFalse(require_stacking)
    format = RemoteBzrDirFormat()
    format._network_name = bzrdir_name
    self._supply_sub_formats_to(format)
    bzrdir = RemoteBzrDir(transport, format, _client=client)
    if repo_path:
        repo_format = response_tuple_to_repo_format(response[1:])
        if repo_path == b'.':
            repo_path = b''
        repo_path = repo_path.decode('utf-8')
        if repo_path:
            repo_bzrdir_format = RemoteBzrDirFormat()
            repo_bzrdir_format._network_name = response[5]
            repo_bzr = RemoteBzrDir(transport.clone(repo_path), repo_bzrdir_format)
        else:
            repo_bzr = bzrdir
        final_stack = response[8] or None
        if final_stack:
            final_stack = final_stack.decode('utf-8')
        final_stack_pwd = response[9] or None
        if final_stack_pwd:
            final_stack_pwd = urlutils.join(transport.base, final_stack_pwd.decode('utf-8'))
        remote_repo = RemoteRepository(repo_bzr, repo_format)
        if len(response) > 10:
            repo_lock_token = response[10] or None
            remote_repo.lock_write(repo_lock_token, _skip_rpc=True)
            if repo_lock_token:
                remote_repo.dont_leave_lock_in_place()
        else:
            remote_repo.lock_write()
        policy = _mod_bzrdir.UseExistingRepository(remote_repo, final_stack, final_stack_pwd, require_stacking)
        policy.acquire_repository()
    else:
        remote_repo = None
        policy = None
    bzrdir._format.set_branch_format(self.get_branch_format())
    if require_stacking:
        bzrdir._format.require_stacking(_skip_repo=True)
    return (remote_repo, bzrdir, require_stacking, policy)