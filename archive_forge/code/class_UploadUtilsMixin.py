import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
class UploadUtilsMixin:
    """Helper class to write upload tests.

    This class provides helpers to simplify test writing. The emphasis is on
    easy test writing, so each tree modification is committed. This doesn't
    preclude writing tests spawning several revisions to upload more complex
    changes.
    """
    upload_dir = 'upload'
    branch_dir = 'branch'

    def make_branch_and_working_tree(self):
        t = transport.get_transport(self.branch_dir)
        t.ensure_base()
        branch = controldir.ControlDir.create_branch_convenience(t.base, format=controldir.format_registry.make_controldir('default'), force_new_tree=False)
        self.tree = branch.controldir.create_workingtree()
        self.tree.commit('initial empty tree')

    def assertUpFileEqual(self, content, path, base=upload_dir):
        self.assertFileEqual(content, osutils.pathjoin(base, path))

    def assertUpPathModeEqual(self, path, expected_mode, base=upload_dir):
        full_path = osutils.pathjoin(base, path)
        st = os.stat(full_path)
        mode = st.st_mode & 511
        if expected_mode == mode:
            return
        raise AssertionError('For path %s, mode is %s not %s' % (full_path, oct(mode), oct(expected_mode)))

    def assertUpPathDoesNotExist(self, path, base=upload_dir):
        self.assertPathDoesNotExist(osutils.pathjoin(base, path))

    def assertUpPathExists(self, path, base=upload_dir):
        self.assertPathExists(osutils.pathjoin(base, path))

    def set_file_content(self, path, content, base=branch_dir):
        with open(osutils.pathjoin(base, path), 'wb') as f:
            f.write(content)

    def add_file(self, path, content, base=branch_dir):
        self.set_file_content(path, content, base)
        self.tree.add(path)
        self.tree.commit('add file %s' % path)

    def modify_file(self, path, content, base=branch_dir):
        self.set_file_content(path, content, base)
        self.tree.commit('modify file %s' % path)

    def chmod_file(self, path, mode, base=branch_dir):
        full_path = osutils.pathjoin(base, path)
        os.chmod(full_path, mode)
        self.tree.commit('change file {} mode to {}'.format(path, oct(mode)))

    def delete_any(self, path, base=branch_dir):
        self.tree.remove([path], keep_files=False)
        self.tree.commit('delete %s' % path)

    def add_dir(self, path, base=branch_dir):
        os.mkdir(osutils.pathjoin(base, path))
        self.tree.add(path)
        self.tree.commit('add directory %s' % path)

    def rename_any(self, old_path, new_path):
        self.tree.rename_one(old_path, new_path)
        self.tree.commit('rename {} into {}'.format(old_path, new_path))

    def transform_dir_into_file(self, path, content, base=branch_dir):
        osutils.delete_any(osutils.pathjoin(base, path))
        self.set_file_content(path, content, base)
        self.tree.commit('change %s from dir to file' % path)

    def transform_file_into_dir(self, path, base=branch_dir):
        self.tree.remove([path], keep_files=False)
        os.mkdir(osutils.pathjoin(base, path))
        self.tree.add(path)
        self.tree.commit('change %s from file to dir' % path)

    def add_symlink(self, path, target, base=branch_dir):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        os.symlink(target, osutils.pathjoin(base, path))
        self.tree.add(path)
        self.tree.commit('add symlink {} -> {}'.format(path, target))

    def modify_symlink(self, path, target, base=branch_dir):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        full_path = osutils.pathjoin(base, path)
        os.unlink(full_path)
        os.symlink(target, full_path)
        self.tree.commit('modify symlink {} -> {}'.format(path, target))

    def _get_cmd_upload(self):
        cmd = cmds.cmd_upload()
        cmd.outf = sys.stdout
        return cmd

    def do_full_upload(self, *args, **kwargs):
        upload = self._get_cmd_upload()
        up_url = self.get_url(self.upload_dir)
        if kwargs.get('directory', None) is None:
            kwargs['directory'] = self.branch_dir
        kwargs['full'] = True
        kwargs['quiet'] = True
        upload.run(up_url, *args, **kwargs)

    def do_incremental_upload(self, *args, **kwargs):
        upload = self._get_cmd_upload()
        up_url = self.get_url(self.upload_dir)
        if kwargs.get('directory', None) is None:
            kwargs['directory'] = self.branch_dir
        kwargs['quiet'] = True
        upload.run(up_url, *args, **kwargs)