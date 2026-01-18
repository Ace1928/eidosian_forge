import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
class CurrentModuleFilter(StackTraceFilter):
    """Filters stack frames from the module where this is used (best effort)."""

    def __init__(self):
        super().__init__()
        filter_filename = None
        outer_f = None
        f = inspect.currentframe()
        try:
            if f is not None:
                outer_f = f.f_back
                if outer_f is not None:
                    filter_filename = inspect.getsourcefile(outer_f)
            self._filename = filter_filename
            self._cached_set = None
        finally:
            del f
            del outer_f

    def get_filtered_filenames(self):
        if self._cached_set is not None:
            return self._cached_set
        filtered_filenames = frozenset((self._filename,))
        if self.parent is not None:
            filtered_filenames |= self.parent.get_filtered_filenames()
        self._cached_set = filtered_filenames
        return filtered_filenames