import logging
from functools import partial
from transitions import Transition
from ..core import listify
from .markup import MarkupMachine, HierarchicalMarkupMachine
from .nesting import NestedTransition
class GraphMachine(MarkupMachine):
    """ Extends transitions.core.Machine with graph support.
        Is also used as a mixin for HierarchicalMachine.
        Attributes:
            _pickle_blacklist (list): Objects that should not/do not need to be pickled.
            transition_cls (cls): TransitionGraphSupport
    """
    _pickle_blacklist = ['model_graphs']
    transition_cls = TransitionGraphSupport
    machine_attributes = {'directed': 'true', 'strict': 'false', 'rankdir': 'LR'}
    hierarchical_machine_attributes = {'rankdir': 'TB', 'rank': 'source', 'nodesep': '1.5', 'compound': 'true'}
    style_attributes = {'node': {'': {}, 'default': {'style': 'rounded, filled', 'shape': 'rectangle', 'fillcolor': 'white', 'color': 'black', 'peripheries': '1'}, 'inactive': {'fillcolor': 'white', 'color': 'black', 'peripheries': '1'}, 'parallel': {'shape': 'rectangle', 'color': 'black', 'fillcolor': 'white', 'style': 'dashed, rounded, filled', 'peripheries': '1'}, 'active': {'color': 'red', 'fillcolor': 'darksalmon', 'peripheries': '2'}, 'previous': {'color': 'blue', 'fillcolor': 'azure2', 'peripheries': '1'}}, 'edge': {'': {}, 'default': {'color': 'black'}, 'previous': {'color': 'blue'}}, 'graph': {'': {}, 'default': {'color': 'black', 'fillcolor': 'white', 'style': 'solid'}, 'previous': {'color': 'blue', 'fillcolor': 'azure2', 'style': 'filled'}, 'active': {'color': 'red', 'fillcolor': 'darksalmon', 'style': 'filled'}, 'parallel': {'color': 'black', 'fillcolor': 'white', 'style': 'dotted'}}}

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._pickle_blacklist}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model_graphs = {}
        for model in self.models:
            try:
                _ = self._get_graph(model)
            except AttributeError as err:
                _LOGGER.warning('Graph for model could not be initialized after pickling: %s', err)

    def __init__(self, model=MarkupMachine.self_literal, states=None, initial='initial', transitions=None, send_event=False, auto_transitions=True, ordered_transitions=False, ignore_invalid_triggers=None, before_state_change=None, after_state_change=None, name=None, queued=False, prepare_event=None, finalize_event=None, model_attribute='state', on_exception=None, title='State Machine', show_conditions=False, show_state_attributes=False, show_auto_transitions=False, use_pygraphviz=True, **kwargs):
        self.title = title
        self.show_conditions = show_conditions
        self.show_state_attributes = show_state_attributes
        kwargs['auto_transitions_markup'] = show_auto_transitions
        self.model_graphs = {}
        self.graph_cls = self._init_graphviz_engine(use_pygraphviz)
        _LOGGER.debug('Using graph engine %s', self.graph_cls)
        super(GraphMachine, self).__init__(model=model, states=states, initial=initial, transitions=transitions, send_event=send_event, auto_transitions=auto_transitions, ordered_transitions=ordered_transitions, ignore_invalid_triggers=ignore_invalid_triggers, before_state_change=before_state_change, after_state_change=after_state_change, name=name, queued=queued, prepare_event=prepare_event, finalize_event=finalize_event, model_attribute=model_attribute, on_exception=on_exception, **kwargs)
        if not hasattr(self, 'get_graph'):
            setattr(self, 'get_graph', self.get_combined_graph)

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

    def get_combined_graph(self, title=None, force_new=False, show_roi=False):
        """ This method is currently equivalent to 'get_graph' of the first machine's model.
        In future releases of transitions, this function will return a combined graph with active states
        of all models.
        Args:
            title (str): Title of the resulting graph.
            force_new (bool): Whether a new graph should be generated even if another graph already exists. This should
            be true whenever the model's state or machine's transitions/states/events have changed.
            show_roi (bool): If set to True, only render states that are active and/or can be reached from
                the current state.
        Returns: AGraph (pygraphviz) or Digraph (graphviz) graph instance that can be drawn.
        """
        _LOGGER.info('Returning graph of the first model. In future releases, this method will return a combined graph of all models.')
        return self._get_graph(self.models[0], title, force_new, show_roi)

    def add_model(self, model, initial=None):
        models = listify(model)
        super(GraphMachine, self).add_model(models, initial)
        for mod in models:
            mod = self if mod is self.self_literal else mod
            if hasattr(mod, 'get_graph'):
                raise AttributeError('Model already has a get_graph attribute. Graph retrieval cannot be bound.')
            setattr(mod, 'get_graph', partial(self._get_graph, mod))
            _ = mod.get_graph(title=self.title, force_new=True)

    def add_states(self, states, on_enter=None, on_exit=None, ignore_invalid_triggers=None, **kwargs):
        """ Calls the base method and regenerates all models's graphs. """
        super(GraphMachine, self).add_states(states, on_enter=on_enter, on_exit=on_exit, ignore_invalid_triggers=ignore_invalid_triggers, **kwargs)
        for model in self.models:
            model.get_graph(force_new=True)

    def add_transition(self, trigger, source, dest, conditions=None, unless=None, before=None, after=None, prepare=None, **kwargs):
        """ Calls the base method and regenerates all models's graphs. """
        super(GraphMachine, self).add_transition(trigger, source, dest, conditions=conditions, unless=unless, before=before, after=after, prepare=prepare, **kwargs)
        for model in self.models:
            model.get_graph(force_new=True)