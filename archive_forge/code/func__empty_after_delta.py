from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _empty_after_delta(self, delta, candidates):
    new_inv = self._get_proposed_inventory(delta)
    result = []
    for dir in candidates:
        file_id = new_inv.path2id(dir)
        if file_id is None:
            continue
        ie = new_inv.get_entry(file_id)
        if ie.kind != 'directory':
            continue
        if len(ie.children) == 0:
            result.append((dir, file_id))
            if self.verbose:
                self.note('pruning empty directory {}'.format(dir))
    return result