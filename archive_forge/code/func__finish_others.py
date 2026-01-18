from . import errors, osutils
def _finish_others(self):
    """Finish walking the other iterators, so we get all entries."""
    for idx, info in enumerate(self._other_entries):
        other_extra = self._others_extra[idx]
        other_has_more, other_path, other_ie = info
        while other_has_more:
            other_file_id = other_ie.file_id
            if other_file_id not in self._out_of_order_processed:
                other_extra[other_file_id] = (other_path, other_ie)
            other_has_more, other_path, other_ie = self._step_one(self._other_walkers[idx])
    del self._other_entries