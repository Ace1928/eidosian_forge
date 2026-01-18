import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestBzrDirInitializeEx(SmartServerRequestBzrDir):

    def parse_NoneTrueFalse(self, arg):
        if not arg:
            return None
        if arg == b'False':
            return False
        if arg == b'True':
            return True
        raise AssertionError('invalid arg %r' % arg)

    def parse_NoneBytestring(self, arg):
        return arg or None

    def parse_NoneString(self, arg):
        if not arg:
            return None
        return arg.decode('utf-8')

    def _serialize_NoneTrueFalse(self, arg):
        if arg is False:
            return b'False'
        if not arg:
            return b''
        return b'True'

    def do(self, bzrdir_network_name, path, use_existing_dir, create_prefix, force_new_repo, stacked_on, stack_on_pwd, repo_format_name, make_working_trees, shared_repo):
        """Initialize a bzrdir at path as per
        BzrDirFormat.initialize_on_transport_ex.

        New in 1.16.  (Replaces BzrDirFormat.initialize_ex verb from 1.15).

        :return: return SuccessfulSmartServerResponse((repo_path, rich_root,
            tree_ref, external_lookup, repo_network_name,
            repo_bzrdir_network_name, bzrdir_format_network_name,
            NoneTrueFalse(stacking), final_stack, final_stack_pwd,
            repo_lock_token))
        """
        target_transport = self.transport_from_client_path(path)
        format = network_format_registry.get(bzrdir_network_name)
        use_existing_dir = self.parse_NoneTrueFalse(use_existing_dir)
        create_prefix = self.parse_NoneTrueFalse(create_prefix)
        force_new_repo = self.parse_NoneTrueFalse(force_new_repo)
        stacked_on = self.parse_NoneString(stacked_on)
        stack_on_pwd = self.parse_NoneString(stack_on_pwd)
        make_working_trees = self.parse_NoneTrueFalse(make_working_trees)
        shared_repo = self.parse_NoneTrueFalse(shared_repo)
        if stack_on_pwd == b'.':
            stack_on_pwd = target_transport.base.encode('utf-8')
        repo_format_name = self.parse_NoneBytestring(repo_format_name)
        repo, bzrdir, stacking, repository_policy = format.initialize_on_transport_ex(target_transport, use_existing_dir=use_existing_dir, create_prefix=create_prefix, force_new_repo=force_new_repo, stacked_on=stacked_on, stack_on_pwd=stack_on_pwd, repo_format_name=repo_format_name, make_working_trees=make_working_trees, shared_repo=shared_repo)
        if repo is None:
            repo_path = ''
            repo_name = b''
            rich_root = tree_ref = external_lookup = b''
            repo_bzrdir_name = b''
            final_stack = None
            final_stack_pwd = None
            repo_lock_token = b''
        else:
            repo_path = self._repo_relpath(bzrdir.root_transport, repo)
            if repo_path == '':
                repo_path = '.'
            rich_root, tree_ref, external_lookup = self._format_to_capabilities(repo._format)
            repo_name = repo._format.network_name()
            repo_bzrdir_name = repo.controldir._format.network_name()
            final_stack = repository_policy._stack_on
            final_stack_pwd = repository_policy._stack_on_pwd
            repo.unlock()
            repo_lock_token = repo.lock_write().repository_token or b''
            if repo_lock_token:
                repo.leave_lock_in_place()
            repo.unlock()
        final_stack = final_stack or ''
        final_stack_pwd = final_stack_pwd or ''
        if final_stack_pwd:
            final_stack_pwd = urlutils.relative_url(target_transport.base, final_stack_pwd)
        if final_stack.startswith('/'):
            client_path = self._root_client_path + final_stack[1:]
            final_stack = urlutils.relative_url(self._root_client_path, client_path)
            final_stack_pwd = '.'
        return SuccessfulSmartServerResponse((repo_path.encode('utf-8'), rich_root, tree_ref, external_lookup, repo_name, repo_bzrdir_name, bzrdir._format.network_name(), self._serialize_NoneTrueFalse(stacking), final_stack.encode('utf-8'), final_stack_pwd.encode('utf-8'), repo_lock_token))