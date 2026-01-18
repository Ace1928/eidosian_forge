from . import errors, osutils
def _walk_master_tree(self):
    """First pass, walk all trees in lock-step.

        When we are done, all nodes in the master_tree will have been
        processed. _other_walkers, _other_entries, and _others_extra will be
        set on 'self' for future processing.
        """
    master_iterator = self._master_tree.iter_entries_by_dir()
    other_walkers = [other.iter_entries_by_dir() for other in self._other_trees]
    other_entries = [self._step_one(walker) for walker in other_walkers]
    others_extra = [{} for _ in range(len(self._other_trees))]
    master_has_more = True
    step_one = self._step_one
    lookup_by_file_id = self._lookup_by_file_id
    out_of_order_processed = self._out_of_order_processed
    while master_has_more:
        master_has_more, path, master_ie = step_one(master_iterator)
        if not master_has_more:
            break
        other_values = []
        other_values_append = other_values.append
        next_other_entries = []
        next_other_entries_append = next_other_entries.append
        for idx, (other_has_more, other_path, other_ie) in enumerate(other_entries):
            if not other_has_more:
                other_values_append(self._lookup_by_master_path(others_extra[idx], self._other_trees[idx], path))
                next_other_entries_append((False, None, None))
            elif master_ie.file_id == other_ie.file_id:
                other_values_append((other_path, other_ie))
                next_other_entries_append(step_one(other_walkers[idx]))
            else:
                other_walker = other_walkers[idx]
                other_extra = others_extra[idx]
                while other_has_more and self._lt_path_by_dirblock(other_path, path):
                    other_file_id = other_ie.file_id
                    if other_file_id not in out_of_order_processed:
                        other_extra[other_file_id] = (other_path, other_ie)
                    other_has_more, other_path, other_ie = step_one(other_walker)
                if other_has_more and other_ie.file_id == master_ie.file_id:
                    other_values_append((other_path, other_ie))
                    other_has_more, other_path, other_ie = step_one(other_walker)
                else:
                    other_values_append(self._lookup_by_master_path(other_extra, self._other_trees[idx], path))
                next_other_entries_append((other_has_more, other_path, other_ie))
        other_entries = next_other_entries
        yield (path, master_ie.file_id, master_ie, other_values)
    self._other_walkers = other_walkers
    self._other_entries = other_entries
    self._others_extra = others_extra