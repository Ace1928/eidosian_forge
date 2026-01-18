from . import errors
from . import graph as _mod_graph
from . import osutils, ui
def _update_from_first_parent(self, key, annotations, lines, parent_key):
    """Reannotate this text relative to its first parent."""
    parent_annotations, matching_blocks = self._get_parent_annotations_and_matches(key, lines, parent_key)
    for parent_idx, lines_idx, match_len in matching_blocks:
        annotations[lines_idx:lines_idx + match_len] = parent_annotations[parent_idx:parent_idx + match_len]