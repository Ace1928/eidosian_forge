from traits.observation._has_traits_helpers import (
from traits.observation._i_observer import IObserver
from traits.observation._observe import add_or_remove_notifiers
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation._trait_change_event import trait_event_factory
@IObserver.register
class TraitAddedObserver:
    """ An observer for observing the trait_added event.

    This observer only offers the "maintainer". When this observer is used in
    an ObserverGraph, its subgraphs are the graphs to be hooked up when a new
    trait is added, provided that the trait satisfies a given criterion.
    The criterion should align with the root observer(s) in these subgraph(s).

    Parameters
    ----------
    match_func : callable(str, CTrait) -> bool
        A callable that receives the name of the trait added and the
        corresponding trait. The returned boolean indicates whether
        notifiers should be added/removed for the added trait.
        This callable is used for equality check and must be hashable.
    optional : boolean
        Whether to skip this observer if the trait_added trait cannot be
        found on the incoming object.
    """

    def __init__(self, match_func, optional):
        self.match_func = match_func
        self.optional = optional

    def __hash__(self):
        """ Return a hash of this object."""
        return hash((type(self).__name__, self.match_func, self.optional))

    def __eq__(self, other):
        """ Return true if this observer is equal to the given one."""
        return type(self) is type(other) and self.match_func == other.match_func and (self.optional == other.optional)

    @property
    def notify(self):
        """ A boolean for whether this observer will notify for changes.
        """
        return False

    def iter_observables(self, object):
        """ Yield observables for notifiers to be attached to or detached from.

        Parameters
        ----------
        object: object
            Object provided by the ``iter_objects`` methods from another
            observers or directly by the user.

        Yields
        ------
        IObservable

        Raises
        ------
        ValueError
            If the given object cannot be handled by this observer.
        """
        if not object_has_named_trait(object, 'trait_added'):
            if self.optional:
                return
            raise ValueError("Unable to observe 'trait_added' event on {!r}".format(object))
        yield object._trait('trait_added', 2)

    def iter_objects(self, object):
        """ Yield objects for the next observer following this observer, in an
        ObserverGraph.

        Parameters
        ----------
        object: object
            Object provided by the ``iter_objects`` methods from another
            observers or directly by the user.

        Yields
        ------
        value : object
        """
        yield from ()

    def get_maintainer(self, graph, handler, target, dispatcher):
        """ Return a notifier for maintaining downstream observers when
        trait_added event happens.

        Parameters
        ----------
        graph : ObserverGraph
            Description for the *downstream* observers, i.e. excluding self.
        handler : callable
            User handler.
        target : object
            Object seen by the user as the owner of the observer.
        dispatcher : callable
            Callable for dispatching the handler.

        Returns
        -------
        notifier : ObserverChangeNotifier
        """
        return ObserverChangeNotifier(observer_handler=self.observer_change_handler, event_factory=trait_event_factory, prevent_event=self.prevent_event, graph=graph, handler=handler, target=target, dispatcher=dispatcher)

    def prevent_event(self, event):
        """ Return true if the added trait should not be handled.

        Parameters
        ----------
        event : TraitChangeEvent
            Event triggered by add_trait.
        """
        object = event.object
        name = event.new
        trait = object.trait(name=name)
        return not self.match_func(name, trait)

    @staticmethod
    def observer_change_handler(event, graph, handler, target, dispatcher):
        """ Handler for maintaining observers.

        Parameters
        ----------
        event : TraitChangeEvent
            Event triggered by add_trait.
        graph : ObserverGraph
            Description for the *downstream* observers, i.e. excluding self.
        handler : callable
            User handler.
        target : object
            Object seen by the user as the owner of the observer.
        dispatcher : callable
            Callable for dispatching the handler.
        """
        new_graph = ObserverGraph(node=_RestrictedNamedTraitObserver(name=event.new, wrapped_observer=graph.node), children=graph.children)
        add_or_remove_notifiers(object=event.object, graph=new_graph, handler=handler, target=target, dispatcher=dispatcher, remove=False)

    def iter_extra_graphs(self, graph):
        """ Yield new ObserverGraph to be contributed by this observer.

        Parameters
        ----------
        graph : ObserverGraph
            The graph this observer is part of.

        Yields
        ------
        ObserverGraph
        """
        yield from ()