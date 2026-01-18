import logging
from functools import partial
from transitions import Transition
from ..core import listify
from .markup import MarkupMachine, HierarchicalMarkupMachine
from .nesting import NestedTransition
class HierarchicalGraphMachine(GraphMachine, HierarchicalMarkupMachine):
    """
        A hierarchical state machine with graph support.
    """
    transition_cls = NestedGraphTransition