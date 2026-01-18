import collections
import copy
import gc
from tensorflow.python.framework import _python_memory_checker_helper
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
class _PythonMemoryChecker(object):
    """Python memory leak detection class."""

    def __init__(self):
        self._snapshots = []

        def _record_snapshot():
            self._snapshots.append(_create_python_object_snapshot())
        self._record_snapshot = _record_snapshot

    def record_snapshot(self):
        _python_memory_checker_helper.mark_stack_trace_and_call(self._record_snapshot)

    @trace.trace_wrapper
    def report(self):
        pass

    @trace.trace_wrapper
    def assert_no_leak_if_all_possibly_except_one(self):
        """Raises an exception if a leak is detected.

    This algorithm classifies a series of allocations as a leak if it's the same
    type at every snapshot, but possibly except one snapshot.
    """
        snapshot_diffs = []
        for i in range(0, len(self._snapshots) - 1):
            snapshot_diffs.append(self._snapshot_diff(i, i + 1))
        allocation_counter = collections.Counter()
        for diff in snapshot_diffs:
            for name, count in diff.items():
                if count > 0:
                    allocation_counter[name] += 1
        leaking_object_names = {name for name, count in allocation_counter.items() if count >= len(snapshot_diffs) - 1}
        if leaking_object_names:
            object_list_to_print = '\n'.join([' - ' + name for name in leaking_object_names])
            raise AssertionError(f'These Python objects were allocated in every snapshot possibly except one.\n\n{object_list_to_print}')

    @trace.trace_wrapper
    def assert_no_new_objects(self, threshold=None):
        """Assert no new Python objects."""
        if not threshold:
            threshold = {}
        count_diff = self._snapshot_diff(0, -1)
        original_count_diff = copy.deepcopy(count_diff)
        count_diff.subtract(collections.Counter(threshold))
        if max(count_diff.values() or [0]) > 0:
            raise AssertionError(f'New Python objects created exceeded the threshold.\nPython object threshold:\n{threshold}\n\nNew Python objects:\n{original_count_diff.most_common()}')
        elif min(count_diff.values(), default=0) < 0:
            logging.warning(f'New Python objects created were less than the threshold.\nPython object threshold:\n{threshold}\n\nNew Python objects:\n{original_count_diff.most_common()}')

    @trace.trace_wrapper
    def _snapshot_diff(self, old_index, new_index):
        return _snapshot_diff(self._snapshots[old_index], self._snapshots[new_index], self._get_internal_object_ids())

    @trace.trace_wrapper
    def _get_internal_object_ids(self):
        ids = set()
        for snapshot in self._snapshots:
            ids.add(id(snapshot))
            for v in snapshot.values():
                ids.add(id(v))
        return ids