from collections import deque
from threading import Event
from kombu.common import ignore_errors
from kombu.utils.encoding import bytes_to_str
from kombu.utils.imports import symbol_by_name
from .utils.graph import DependencyGraph, GraphFormatter
from .utils.imports import instantiate, qualname
from .utils.log import get_logger
class StepFormatter(GraphFormatter):
    """Graph formatter for :class:`Blueprint`."""
    blueprint_prefix = '⧉'
    conditional_prefix = '∘'
    blueprint_scheme = {'shape': 'parallelogram', 'color': 'slategray4', 'fillcolor': 'slategray3'}

    def label(self, step):
        return step and '{}{}'.format(self._get_prefix(step), bytes_to_str((step.label or _label(step)).encode('utf-8', 'ignore')))

    def _get_prefix(self, step):
        if step.last:
            return self.blueprint_prefix
        if step.conditional:
            return self.conditional_prefix
        return ''

    def node(self, obj, **attrs):
        scheme = self.blueprint_scheme if obj.last else self.node_scheme
        return self.draw_node(obj, scheme, attrs)

    def edge(self, a, b, **attrs):
        if a.last:
            attrs.update(arrowhead='none', color='darkseagreen3')
        return self.draw_edge(a, b, self.edge_scheme, attrs)