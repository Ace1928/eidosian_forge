from breezy.errors import BzrError, DependencyNotPresent
from breezy.branch import Branch
def _delta_files(tree_delta):
    for path, file_id, kind in tree_delta.added:
        if kind == 'file':
            yield path
    for path, file_id, kind, text_modified, meta_modified in tree_delta.modified:
        if kind == 'file' and text_modified:
            yield path
    for oldpath, newpath, id, kind, text_modified, meta_modified in tree_delta.renamed:
        if kind == 'file' and text_modified:
            yield newpath
    for path, id, old_kind, new_kind in tree_delta.kind_changed:
        if new_kind == 'file':
            yield path