from traits.observation._dict_change_event import dict_event_factory
from traits.observation._i_observer import IObserver
from traits.observation._observe import add_or_remove_notifiers
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._trait_event_notifier import TraitEventNotifier
from traits.trait_dict_object import TraitDict
 Yield new ObserverGraph to be contributed by this observer.

        Parameters
        ----------
        graph : ObserverGraph
            The graph this observer is part of.

        Yields
        ------
        ObserverGraph
        