from . import errors
from . import graph as _mod_graph
from . import osutils, ui
def _record_annotation(self, key, parent_keys, annotations):
    self._annotations_cache[key] = annotations
    for parent_key in parent_keys:
        num = self._num_needed_children[parent_key]
        num -= 1
        if num == 0:
            del self._text_cache[parent_key]
            del self._annotations_cache[parent_key]
        self._num_needed_children[parent_key] = num