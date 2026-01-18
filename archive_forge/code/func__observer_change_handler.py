from traits.observation._list_change_event import list_event_factory
from traits.observation._i_observer import IObserver
from traits.observation._trait_event_notifier import TraitEventNotifier
from traits.observation._observe import add_or_remove_notifiers
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.trait_list_object import TraitList
def _observer_change_handler(event, graph, handler, target, dispatcher):
    """ Handler for maintaining observers. Used by ObserverChangeNotifier.

    The downstream notifiers are removed from items removed from the list.
    Likewise, downstream notifiers are added to items added to the list.

    Parameters
    ----------
    event : ListChangeEvent
        Change event that triggers the maintainer.
    graph : ObserverGraph
        Description for the *downstream* observers, i.e. excluding self.
    handler : callable
        User handler.
    target : object
        Object seen by the user as the owner of the observer.
    dispatcher : callable
        Callable for dispatching the handler.
    """
    for removed_item in event.removed:
        add_or_remove_notifiers(object=removed_item, graph=graph, handler=handler, target=target, dispatcher=dispatcher, remove=True)
    for added_item in event.added:
        add_or_remove_notifiers(object=added_item, graph=graph, handler=handler, target=target, dispatcher=dispatcher, remove=False)