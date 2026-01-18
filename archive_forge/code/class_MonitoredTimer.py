import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
class MonitoredTimer(object):
    """A context manager to measure the walltime and increment a Counter cell."""
    __slots__ = ['cell', 't', 'monitored_section_name', '_counting', '_avoid_repetitive_counting']

    def __init__(self, cell, monitored_section_name=None, avoid_repetitive_counting=False):
        """Creates a new MonitoredTimer.

    Args:
      cell: the cell associated with the time metric that will be inremented.
      monitored_section_name: name of action being monitored here.
      avoid_repetitive_counting: when set to True, if already in a monitored
        timer section with the same monitored_section_name, skip counting.
    """
        self.cell = cell
        self.monitored_section_name = monitored_section_name
        self._avoid_repetitive_counting = avoid_repetitive_counting
        self._counting = True

    def __enter__(self):
        if self._avoid_repetitive_counting and self.monitored_section_name and (self.monitored_section_name in MonitoredTimerSections):
            self._counting = False
            return self
        self.t = time.time()
        if self.monitored_section_name:
            MonitoredTimerSections.append(self.monitored_section_name)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        del exception_type, exception_value, traceback
        if self._counting:
            micro_seconds = (time.time() - self.t) * 1000000
            self.cell.increase_by(int(micro_seconds))
            if self.monitored_section_name:
                MonitoredTimerSections.remove(self.monitored_section_name)