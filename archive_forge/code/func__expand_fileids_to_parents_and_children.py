from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _expand_fileids_to_parents_and_children(self, file_ids):
    """Give a more wholistic view starting with the given file_ids.

        For any file_id which maps to a directory, we will include all children
        of that directory. We will also include all directories which are
        parents of the given file_ids, but we will not include their children.

        eg:
          /     # TREE_ROOT
          foo/  # foo-id
            baz # baz-id
            frob/ # frob-id
              fringle # fringle-id
          bar/  # bar-id
            bing # bing-id

        if given [foo-id] we will include
            TREE_ROOT as interesting parents
        and
            foo-id, baz-id, frob-id, fringle-id
        As interesting ids.
        """
    interesting = set()
    directories_to_expand = set()
    children_of_parent_id = {}
    for entry in self._getitems(file_ids):
        if entry.kind == 'directory':
            directories_to_expand.add(entry.file_id)
        interesting.add(entry.parent_id)
        children_of_parent_id.setdefault(entry.parent_id, set()).add(entry.file_id)
    remaining_parents = interesting.difference(file_ids)
    interesting.add(None)
    remaining_parents.discard(None)
    while remaining_parents:
        next_parents = set()
        for entry in self._getitems(remaining_parents):
            next_parents.add(entry.parent_id)
            children_of_parent_id.setdefault(entry.parent_id, set()).add(entry.file_id)
        remaining_parents = next_parents.difference(interesting)
        interesting.update(remaining_parents)
    interesting.update(file_ids)
    interesting.discard(None)
    while directories_to_expand:
        keys = [StaticTuple(f).intern() for f in directories_to_expand]
        directories_to_expand = set()
        items = self.parent_id_basename_to_file_id.iteritems(keys)
        next_file_ids = {item[1] for item in items}
        next_file_ids = next_file_ids.difference(interesting)
        interesting.update(next_file_ids)
        for entry in self._getitems(next_file_ids):
            if entry.kind == 'directory':
                directories_to_expand.add(entry.file_id)
            children_of_parent_id.setdefault(entry.parent_id, set()).add(entry.file_id)
    return (interesting, children_of_parent_id)