import functools
from traits.observation._anytrait_filter import anytrait_filter
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._observer_graph import ObserverGraph
from traits.observation._set_item_observer import SetItemObserver
class ParallelObserverExpression(ObserverExpression):
    """ Container of ObserverExpression for joining expressions in parallel.

    Parameters
    ----------
    left : ObserverExpression
        Left expression to be joined in parallel.
    right : ObserverExpression
        Right expression to be joined in parallel.
    """
    __slots__ = ('_left', '_right')

    def __init__(self, left, right):
        self._left = left
        self._right = right

    def __hash__(self):
        return hash((type(self).__name__, self._left, self._right))

    def __eq__(self, other):
        return type(self) is type(other) and self._left == other._left and (self._right == other._right)

    def _create_graphs(self, branches):
        left_graphs = self._left._create_graphs(branches=branches)
        right_graphs = self._right._create_graphs(branches=branches)
        return left_graphs + right_graphs