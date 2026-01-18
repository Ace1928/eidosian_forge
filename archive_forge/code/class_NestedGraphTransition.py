import logging
from functools import partial
from transitions import Transition
from ..core import listify
from .markup import MarkupMachine, HierarchicalMarkupMachine
from .nesting import NestedTransition
class NestedGraphTransition(TransitionGraphSupport, NestedTransition):
    """
        A transition type to be used with (subclasses of) `HierarchicalGraphMachine` and
        `LockedHierarchicalGraphMachine`.
    """