import errno
import os
import posixpath
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from dulwich.index import blob_from_path_and_stat, commit_tree
from dulwich.objects import Blob
from .. import annotate, conflicts, errors, multiparent, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import InterTree, TreeChange
from .mapping import (decode_git_path, encode_git_path, mode_is_executable,
from .tree import GitTree, GitTreeDirectory, GitTreeFile, GitTreeSymlink
def _apply_insertions(self, mover):
    """Perform tree operations that insert directory/inventory names.

        That is, create any files that need to be created, and restore from
        limbo any files that needed renaming.  This must be done in strict
        parent-to-child order.

        If inventory_delta is None, no inventory delta is calculated, and
        no list of modified paths is returned.
        """
    new_paths = self.new_paths(filesystem_only=True)
    modified_paths = []
    with ui.ui_factory.nested_progress_bar() as child_pb:
        for num, (path, trans_id) in enumerate(new_paths):
            if num % 10 == 0:
                child_pb.update(gettext('adding file'), num, len(new_paths))
            full_path = self._tree.abspath(path)
            if trans_id in self._needs_rename:
                try:
                    mover.rename(self._limbo_name(trans_id), full_path)
                except TransformRenameFailed as e:
                    if e.errno != errno.ENOENT:
                        raise
                else:
                    self.rename_count += 1
            if trans_id in self._new_contents or self.path_changed(trans_id):
                if trans_id in self._new_contents:
                    modified_paths.append(full_path)
            if trans_id in self._new_executability:
                self._set_executability(path, trans_id)
            if trans_id in self._observed_sha1s:
                o_sha1, o_st_val = self._observed_sha1s[trans_id]
                st = osutils.lstat(full_path)
                self._observed_sha1s[trans_id] = (o_sha1, st)
            if trans_id in self._new_reference_revision:
                for submodule_path, submodule_url, submodule_name in self._tree._submodule_config():
                    if decode_git_path(submodule_path) == path:
                        break
                else:
                    trace.warning('unable to find submodule for path %s', path)
                    continue
                submodule_transport = self._tree.controldir.control_transport.clone(os.path.join('modules', submodule_name.decode('utf-8')))
                submodule_transport.create_prefix()
                from .dir import BareLocalGitControlDirFormat
                BareLocalGitControlDirFormat().initialize_on_transport(submodule_transport)
                with open(os.path.join(full_path, '.git'), 'w') as f:
                    submodule_abspath = submodule_transport.local_abspath('.')
                    f.write('gitdir: %s\n' % os.path.relpath(submodule_abspath, full_path))
    for path, trans_id in new_paths:
        if trans_id in self._limbo_files:
            del self._limbo_files[trans_id]
    self._new_contents.clear()
    return modified_paths