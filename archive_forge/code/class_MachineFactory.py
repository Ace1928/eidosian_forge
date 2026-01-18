from functools import partial
from ..core import Machine, Transition
from .nesting import HierarchicalMachine, NestedEvent, NestedTransition
from .locking import LockedMachine
from .diagrams import GraphMachine, NestedGraphTransition, HierarchicalGraphMachine
class MachineFactory(object):
    """ Convenience factory for machine class retrieval. """

    @staticmethod
    def get_predefined(graph=False, nested=False, locked=False, asyncio=False):
        """ A function to retrieve machine classes by required functionality.
        Args:
            graph (bool): Whether the returned class should contain graph support.
            nested: Whether the returned machine class should support nested states.
            locked: Whether the returned class should facilitate locks for threadsafety.

        Returns (class): A machine class with the specified features.
        """
        try:
            return _CLASS_MAP[graph, nested, locked, asyncio]
        except KeyError:
            raise ValueError('Feature combination not (yet) supported')