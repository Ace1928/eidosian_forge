from functools import partial
from ..core import Machine, Transition
from .nesting import HierarchicalMachine, NestedEvent, NestedTransition
from .locking import LockedMachine
from .diagrams import GraphMachine, NestedGraphTransition, HierarchicalGraphMachine
class AsyncGraphMachine(GraphMachine, AsyncMachine):
    """ A machine that supports asynchronous event/callback processing with Graphviz support. """
    transition_cls = AsyncTransition