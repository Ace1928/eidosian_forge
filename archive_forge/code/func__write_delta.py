from .... import errors
from .... import transport as _mod_transport
from .... import ui
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....textfile import text_file
from ....timestamp import format_highres_date
from ....trace import mutter
from ...testament import StrictTestament
from ..bundle_data import BundleInfo, RevisionInfo
from . import BundleSerializer, _get_bundle_header, binary_diff
def _write_delta(self, new_tree, old_tree, default_revision_id, force_binary):
    """Write out the changes between the trees."""
    DEVNULL = '/dev/null'
    old_label = ''
    new_label = ''

    def do_diff(file_id, old_path, new_path, action, force_binary):

        def tree_lines(tree, path, require_text=False):
            try:
                tree_file = tree.get_file(path)
            except _mod_transport.NoSuchFile:
                return []
            else:
                if require_text is True:
                    tree_file = text_file(tree_file)
                return tree_file.readlines()
        try:
            if force_binary:
                raise errors.BinaryFile()
            old_lines = tree_lines(old_tree, old_path, require_text=True)
            new_lines = tree_lines(new_tree, new_path, require_text=True)
            action.write(self.to_file)
            internal_diff(old_path, old_lines, new_path, new_lines, self.to_file)
        except errors.BinaryFile:
            old_lines = tree_lines(old_tree, old_path, require_text=False)
            new_lines = tree_lines(new_tree, new_path, require_text=False)
            action.add_property('encoding', 'base64')
            action.write(self.to_file)
            binary_diff(old_path, old_lines, new_path, new_lines, self.to_file)

    def finish_action(action, file_id, kind, meta_modified, text_modified, old_path, new_path):
        entry = new_tree.root_inventory.get_entry(file_id)
        if entry.revision != default_revision_id:
            action.add_utf8_property('last-changed', entry.revision)
        if meta_modified:
            action.add_bool_property('executable', entry.executable)
        if text_modified and kind == 'symlink':
            action.add_property('target', entry.symlink_target)
        if text_modified and kind == 'file':
            do_diff(file_id, old_path, new_path, action, force_binary)
        else:
            action.write(self.to_file)
    delta = new_tree.changes_from(old_tree, want_unchanged=True, include_root=True)
    for change in delta.removed:
        action = Action('removed', [change.kind[0], change.path[0]]).write(self.to_file)
    for change in delta.added + delta.copied:
        action = Action('added', [change.kind[1], change.path[1]], [('file-id', change.file_id.decode('utf-8'))])
        meta_modified = change.kind[1] == 'file' and change.executable[1]
        finish_action(action, change.file_id, change.kind[1], meta_modified, change.changed_content, DEVNULL, change.path[1])
    for change in delta.renamed:
        action = Action('renamed', [change.kind[1], change.path[0]], [(change.path[1],)])
        finish_action(action, change.file_id, change.kind[1], change.meta_modified(), change.changed_content, change.path[0], change.path[1])
    for change in delta.modified:
        action = Action('modified', [change.kind[1], change.path[1]])
        finish_action(action, change.file_id, change.kind[1], change.meta_modified(), change.changed_content, change.path[0], change.path[1])
    for change in delta.unchanged:
        new_rev = new_tree.get_file_revision(change.path[1])
        if new_rev is None:
            continue
        old_rev = old_tree.get_file_revision(change.path[0])
        if new_rev != old_rev:
            action = Action('modified', [change.kind[1], change.path[1]])
            action.add_utf8_property('last-changed', new_rev)
            action.write(self.to_file)