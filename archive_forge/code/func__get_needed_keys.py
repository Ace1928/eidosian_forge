from . import errors
from . import graph as _mod_graph
from . import osutils, ui
def _get_needed_keys(self, key):
    """Determine the texts we need to get from the backing vf.

        :return: (vf_keys_needed, ann_keys_needed)
            vf_keys_needed  These are keys that we need to get from the vf
            ann_keys_needed Texts which we have in self._text_cache but we
                            don't have annotations for. We need to yield these
                            in the proper order so that we can get proper
                            annotations.
        """
    parent_map = self._parent_map
    self._num_needed_children[key] = 1
    vf_keys_needed = set()
    ann_keys_needed = set()
    needed_keys = {key}
    while needed_keys:
        parent_lookup = []
        next_parent_map = {}
        for key in needed_keys:
            if key in self._parent_map:
                if key not in self._text_cache:
                    vf_keys_needed.add(key)
                elif key not in self._annotations_cache:
                    ann_keys_needed.add(key)
                    next_parent_map[key] = self._parent_map[key]
            else:
                parent_lookup.append(key)
                vf_keys_needed.add(key)
        needed_keys = set()
        next_parent_map.update(self._vf.get_parent_map(parent_lookup))
        for key, parent_keys in next_parent_map.items():
            if parent_keys is None:
                parent_keys = ()
                next_parent_map[key] = ()
            self._update_needed_children(key, parent_keys)
            needed_keys.update([key for key in parent_keys if key not in parent_map])
        parent_map.update(next_parent_map)
        self._heads_provider = None
    return (vf_keys_needed, ann_keys_needed)