import time
from io import BytesIO
from dulwich import __version__ as dulwich_version
from dulwich.objects import Blob
from .. import __version__ as brz_version
from .. import branch as _mod_branch
from .. import diff as _mod_diff
from .. import errors, osutils
from .. import revision as _mod_revision
from ..merge_directive import BaseMergeDirective
from .mapping import object_mode
from .object_store import get_object_store
class GitDiffTree(_mod_diff.DiffTree):
    """Provides a text representation between two trees, formatted for svn."""

    def _show_diff(self, specific_files, extra_trees):
        from dulwich.patch import write_blob_diff
        iterator = self.new_tree.iter_changes(self.old_tree, specific_files=specific_files, extra_trees=extra_trees, require_versioned=True)
        has_changes = 0

        def get_encoded_path(path):
            if path is not None:
                return path.encode(self.path_encoding, 'replace')

        def get_file_mode(tree, path, kind, executable):
            if path is None:
                return 0
            return object_mode(kind, executable)

        def get_blob(present, tree, path):
            if present:
                with tree.get_file(path) as f:
                    return Blob.from_string(f.read())
            else:
                return None
        trees = (self.old_tree, self.new_tree)
        for change in iterator:
            if change.parent_id == (None, None) or change.kind == (None, None):
                continue
            path_encoded = (get_encoded_path(change.path[0]), get_encoded_path(change.path[1]))
            present = (change.kind[0] not in (None, 'directory'), change.kind[1] not in (None, 'directory'))
            if not present[0] and (not present[1]):
                continue
            contents = (get_blob(present[0], trees[0], change.path[0]), get_blob(present[1], trees[1], change.path[1]))
            renamed = (change.parent_id[0], change.name[0]) != (change.parent_id[1], change.name[1])
            mode = (get_file_mode(trees[0], path_encoded[0], change.kind[0], change.executable[0]), get_file_mode(trees[1], path_encoded[1], change.kind[1], change.executable[1]))
            write_blob_diff(self.to_file, (path_encoded[0], mode[0], contents[0]), (path_encoded[1], mode[1], contents[1]))
            has_changes |= change.changed_content or renamed
        return has_changes