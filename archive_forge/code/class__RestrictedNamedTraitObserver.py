from traits.observation._has_traits_helpers import (
from traits.observation._i_observer import IObserver
from traits.observation._observe import add_or_remove_notifiers
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation._trait_change_event import trait_event_factory
@IObserver.register
class _RestrictedNamedTraitObserver:
    """ An observer to support TraitAddedObserver in order to add
    notifiers for one specific named trait. The notifiers should be
    contributed by the original observer.

    Parameters
    ----------
    name : str
        Name of the trait to be observed.
    wrapped_observer : IObserver
        The observer from which notifers are obtained.
    """

    def __init__(self, name, wrapped_observer):
        self.name = name
        self._wrapped_observer = wrapped_observer

    def __hash__(self):
        return hash((type(self), self.name, self._wrapped_observer))

    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name and (self._wrapped_observer == other._wrapped_observer)

    @property
    def notify(self):
        """ A boolean for whether this observer will notify for changes. """
        return self._wrapped_observer.notify

    def iter_observables(self, object):
        """ Yield only the observable for the named trait."""
        yield object._trait(self.name, 2)

    def iter_objects(self, object):
        """ Yield only the value for the named trait."""
        yield from iter_objects(object, self.name)

    def get_notifier(self, handler, target, dispatcher):
        """ Return the notifier from the wrapped observer."""
        return self._wrapped_observer.get_notifier(handler, target, dispatcher)

    def get_maintainer(self, graph, handler, target, dispatcher):
        """ Return the maintainer from the wrapped observer."""
        return self._wrapped_observer.get_maintainer(graph, handler, target, dispatcher)

    def iter_extra_graphs(self, graph):
        yield from ()