from collections import deque
from threading import Event
from kombu.common import ignore_errors
from kombu.utils.encoding import bytes_to_str
from kombu.utils.imports import symbol_by_name
from .utils.graph import DependencyGraph, GraphFormatter
from .utils.imports import instantiate, qualname
from .utils.log import get_logger
class StartStopStep(Step):
    """Bootstep that must be started and stopped in order."""
    obj = None

    def start(self, parent):
        if self.obj:
            return self.obj.start()

    def stop(self, parent):
        if self.obj:
            return self.obj.stop()

    def close(self, parent):
        pass

    def terminate(self, parent):
        if self.obj:
            return getattr(self.obj, 'terminate', self.obj.stop)()

    def include(self, parent):
        inc, ret = self._should_include(parent)
        if inc:
            self.obj = ret
            parent.steps.append(self)
        return inc