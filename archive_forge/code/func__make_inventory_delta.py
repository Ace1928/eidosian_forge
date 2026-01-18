from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
def _make_inventory_delta(self, matches):
    delta = []
    file_id_matches = {f: p for p, f in matches.items()}
    file_id_query = []
    for f in matches.values():
        try:
            file_id_query.append(self.tree.id2path(f))
        except errors.NoSuchId:
            pass
    for old_path, entry in self.tree.iter_entries_by_dir(specific_files=file_id_query):
        new_path = file_id_matches[entry.file_id]
        parent_path, new_name = osutils.split(new_path)
        parent_id = matches.get(parent_path)
        if parent_id is None:
            parent_id = self.tree.path2id(parent_path)
            if parent_id is None:
                added, ignored = self.tree.smart_add([parent_path], recurse=False)
                if len(ignored) > 0 and ignored[0] == parent_path:
                    continue
                else:
                    parent_id = self.tree.path2id(parent_path)
        if entry.name == new_name and entry.parent_id == parent_id:
            continue
        new_entry = entry.copy()
        new_entry.parent_id = parent_id
        new_entry.name = new_name
        delta.append((old_path, new_path, new_entry.file_id, new_entry))
    return delta