from collections import deque
from threading import Event
from kombu.common import ignore_errors
from kombu.utils.encoding import bytes_to_str
from kombu.utils.imports import symbol_by_name
from .utils.graph import DependencyGraph, GraphFormatter
from .utils.imports import instantiate, qualname
from .utils.log import get_logger
def _firstpass(self, steps):
    for step in steps.values():
        step.requires = [symbol_by_name(dep) for dep in step.requires]
    stream = deque((step.requires for step in steps.values()))
    while stream:
        for node in stream.popleft():
            node = symbol_by_name(node)
            if node.name not in self.steps:
                steps[node.name] = node
            stream.append(node.requires)