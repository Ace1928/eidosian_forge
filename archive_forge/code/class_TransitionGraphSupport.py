import logging
from functools import partial
from transitions import Transition
from ..core import listify
from .markup import MarkupMachine, HierarchicalMarkupMachine
from .nesting import NestedTransition
class TransitionGraphSupport(Transition):
    """ Transition used in conjunction with (Nested)Graphs to update graphs whenever a transition is
        conducted.
    """

    def __init__(self, *args, **kwargs):
        label = kwargs.pop('label', None)
        super(TransitionGraphSupport, self).__init__(*args, **kwargs)
        if label:
            self.label = label

    def _change_state(self, event_data):
        graph = event_data.machine.model_graphs[id(event_data.model)]
        graph.reset_styling()
        graph.set_previous_transition(self.source, self.dest)
        super(TransitionGraphSupport, self)._change_state(event_data)
        graph = event_data.machine.model_graphs[id(event_data.model)]
        graph.set_node_style(getattr(event_data.model, event_data.machine.model_attribute), 'active')