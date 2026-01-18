import logging
from functools import partial
from transitions import Transition
from ..core import listify
from .markup import MarkupMachine, HierarchicalMarkupMachine
from .nesting import NestedTransition
def _init_graphviz_engine(self, use_pygraphviz):
    """ Imports diagrams (py)graphviz backend based on machine configuration """
    if use_pygraphviz:
        try:
            if hasattr(self.state_cls, 'separator') and hasattr(self, '__enter__'):
                from .diagrams_pygraphviz import NestedGraph as Graph, pgv
                self.machine_attributes.update(self.hierarchical_machine_attributes)
            else:
                from .diagrams_pygraphviz import Graph, pgv
            if pgv is None:
                raise ImportError
            return Graph
        except ImportError:
            _LOGGER.warning('Could not import pygraphviz backend. Will try graphviz backend next')
    if hasattr(self.state_cls, 'separator') and hasattr(self, '__enter__'):
        from .diagrams_graphviz import NestedGraph as Graph
        self.machine_attributes.update(self.hierarchical_machine_attributes)
    else:
        from .diagrams_graphviz import Graph
    return Graph