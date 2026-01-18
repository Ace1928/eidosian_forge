from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
def _find_missing_files(self, basis):
    missing_files = set()
    missing_parents = {}
    candidate_files = set()
    with ui_factory.nested_progress_bar() as task:
        iterator = self.tree.iter_changes(basis, want_unversioned=True, pb=task)
        for change in iterator:
            if change.kind[1] is None and change.versioned[1]:
                if not self.tree.has_filename(self.tree.id2path(change.parent_id[0])):
                    missing_parents.setdefault(change.parent_id[0], set()).add(change.file_id)
                if change.kind[0] == 'file':
                    missing_files.add(change.file_id)
                else:
                    pass
            if change.versioned == (False, False):
                if self.tree.is_ignored(change.path[1]):
                    continue
                if change.kind[1] == 'file':
                    candidate_files.add(change.path[1])
                if change.kind[1] == 'directory':
                    for _dir, children in self.tree.walkdirs(change.path[1]):
                        for child in children:
                            if child[2] == 'file':
                                candidate_files.add(child[0])
    return (missing_files, missing_parents, candidate_files)