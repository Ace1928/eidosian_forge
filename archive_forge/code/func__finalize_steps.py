from collections import deque
from threading import Event
from kombu.common import ignore_errors
from kombu.utils.encoding import bytes_to_str
from kombu.utils.imports import symbol_by_name
from .utils.graph import DependencyGraph, GraphFormatter
from .utils.imports import instantiate, qualname
from .utils.log import get_logger
def _finalize_steps(self, steps):
    last = self._find_last()
    self._firstpass(steps)
    it = ((C, C.requires) for C in steps.values())
    G = self.graph = DependencyGraph(it, formatter=self.GraphFormatter(root=last))
    if last:
        for obj in G:
            if obj != last:
                G.add_edge(last, obj)
    try:
        return G.topsort()
    except KeyError as exc:
        raise KeyError('unknown bootstep: %s' % exc)