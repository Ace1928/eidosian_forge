from traits.observation._has_traits_helpers import (
from traits.observation._i_observer import IObserver
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation._trait_change_event import trait_event_factory
from traits.observation._trait_added_observer import TraitAddedObserver
from traits.observation._trait_event_notifier import TraitEventNotifier
 Yield new ObserverGraph to be contributed by this observer.

        Parameters
        ----------
        graph : ObserverGraph
            The graph this observer is part of.

        Yields
        ------
        ObserverGraph
        