import logging
from functools import partial
from transitions import Transition
from ..core import listify
from .markup import MarkupMachine, HierarchicalMarkupMachine
from .nesting import NestedTransition
def _get_graph(self, model, title=None, force_new=False, show_roi=False):
    """ This method will be bound as a partial to models and return a graph object to be drawn or manipulated.
        Args:
            model (object): The model that `_get_graph` was bound to. This parameter will be set by `GraphMachine`.
            title (str): The title of the created graph.
            force_new (bool): Whether a new graph should be generated even if another graph already exists. This should
            be true whenever the model's state or machine's transitions/states/events have changed.
            show_roi (bool): If set to True, only render states that are active and/or can be reached from
                the current state.
        Returns: AGraph (pygraphviz) or Digraph (graphviz) graph instance that can be drawn.
        """
    if force_new:
        graph = self.graph_cls(self)
        self.model_graphs[id(model)] = graph
        try:
            graph.set_node_style(getattr(model, self.model_attribute), 'active')
        except AttributeError:
            _LOGGER.info('Could not set active state of diagram')
    try:
        graph = self.model_graphs[id(model)]
    except KeyError:
        _ = self._get_graph(model, title, force_new=True)
        graph = self.model_graphs[id(model)]
    return graph.get_graph(title=title, roi_state=getattr(model, self.model_attribute) if show_roi else None)