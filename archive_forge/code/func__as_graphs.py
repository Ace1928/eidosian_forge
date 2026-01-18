import functools
from traits.observation._anytrait_filter import anytrait_filter
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._observer_graph import ObserverGraph
from traits.observation._set_item_observer import SetItemObserver
def _as_graphs(self):
    """ Return all the ObserverGraph for the observer framework to attach
        notifiers.

        This is considered private to the users and to modules outside of the
        ``observation`` subpackage, but public to modules within the
        ``observation`` subpackage.

        Returns
        -------
        graphs : list of ObserverGraph
        """
    return self._create_graphs(branches=[])