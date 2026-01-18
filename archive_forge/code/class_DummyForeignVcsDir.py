from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class DummyForeignVcsDir(bzrdir.BzrDirMeta1):

    def __init__(self, _transport, _format):
        self._format = _format
        self.transport = _transport.clone('.dummy')
        self.root_transport = _transport
        self._mode_check_done = False
        self._control_files = lockable_files.LockableFiles(self.transport, 'lock', lockable_files.TransportLock)

    def create_workingtree(self):
        self.root_transport.put_bytes('.bzr', b'foo')
        return super().create_workingtree()

    def open_branch(self, name=None, unsupported=False, ignore_fallbacks=True, possible_transports=None):
        if name is None:
            name = self._get_selected_branch()
        if name != '':
            raise controldir.NoColocatedBranchSupport(self)
        return self._format.get_branch_format().open(self, _found=True)

    def cloning_metadir(self, stacked=False):
        """Produce a metadir suitable for cloning with."""
        return controldir.format_registry.make_controldir('default')

    def checkout_metadir(self):
        return self.cloning_metadir()

    def sprout(self, url, revision_id=None, force_new_repo=False, recurse='down', possible_transports=None, accelerator_tree=None, hardlink=False, stacked=False, source_branch=None):
        return super().sprout(url=url, revision_id=revision_id, force_new_repo=force_new_repo, recurse=recurse, possible_transports=possible_transports, hardlink=hardlink, stacked=stacked, source_branch=source_branch)