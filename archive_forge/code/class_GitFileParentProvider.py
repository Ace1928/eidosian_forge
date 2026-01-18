import posixpath
import stat
from dulwich.errors import NotTreeError
from dulwich.object_store import tree_lookup_path
from dulwich.objects import SubmoduleEncountered
from ..revision import NULL_REVISION
from .mapping import encode_git_path
class GitFileParentProvider:

    def __init__(self, change_scanner):
        self.change_scanner = change_scanner
        self.store = self.change_scanner.repository._git.object_store

    def _get_parents(self, file_id, text_revision):
        commit_id, mapping = self.change_scanner.repository.lookup_bzr_revision_id(text_revision)
        try:
            path = encode_git_path(mapping.parse_file_id(file_id))
        except ValueError:
            raise KeyError(file_id)
        text_parents = []
        for commit_parent in self.store[commit_id].parents:
            try:
                store, path, text_parent = self.change_scanner.find_last_change_revision(path, commit_parent)
            except KeyError:
                continue
            if text_parent not in text_parents:
                text_parents.append(text_parent)
        return tuple([(file_id, self.change_scanner.repository.lookup_foreign_revision_id(p)) for p in text_parents])

    def get_parent_map(self, keys):
        ret = {}
        for key in keys:
            file_id, text_revision = key
            if text_revision == NULL_REVISION:
                ret[key] = ()
                continue
            if not isinstance(file_id, bytes):
                raise TypeError(file_id)
            if not isinstance(text_revision, bytes):
                raise TypeError(text_revision)
            try:
                ret[key] = self._get_parents(file_id, text_revision)
            except KeyError:
                pass
        return ret