from traits.observation._has_traits_helpers import (
from traits.observation._i_observer import IObserver
from traits.observation._observe import add_or_remove_notifiers
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation._trait_change_event import trait_event_factory
 Return the maintainer from the wrapped observer.