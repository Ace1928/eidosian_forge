import functools
from traits.observation._anytrait_filter import anytrait_filter
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._observer_graph import ObserverGraph
from traits.observation._set_item_observer import SetItemObserver
class ObserverExpression:
    """
    ObserverExpression is an object for describing what traits are being
    observed for change notifications. It can be passed directly to
    ``HasTraits.observe`` method or the ``observe`` decorator.

    An ObserverExpression is typically created using one of the top-level
    functions provided in this module, e.g. ``trait``.
    """
    __slots__ = ()

    def __or__(self, expression):
        """ Create a new expression that matches this expression OR
        the given expression.

        e.g. ``trait("age") | trait("number")`` will match either trait
        **age** or trait **number** on an object.

        Parameters
        ----------
        expression : ObserverExpression

        Returns
        -------
        new_expression : ObserverExpression
        """
        return ParallelObserverExpression(self, expression)

    def then(self, expression):
        """ Create a new expression by extending this expression with
        the given expression.

        e.g. ``trait("child").then( trait("age") | trait("number") )``
        on an object matches ``child.age`` or ``child.number`` on the object.

        Parameters
        ----------
        expression : ObserverExpression

        Returns
        -------
        new_expression : ObserverExpression
        """
        return SeriesObserverExpression(self, expression)

    def match(self, filter, notify=True):
        """ Create a new expression for observing traits using the
        given filter.

        Events emitted (if any) will be instances of
        :class:`~traits.observation.events.TraitChangeEvent`.

        Parameters
        ----------
        filter : callable(str, CTrait) -> bool
            A callable that receives the name of a trait and the corresponding
            trait definition. The returned bool indicates whether the trait
            is observed. In order to remove an existing observer with the
            equivalent filter, the filter callables must compare equally. The
            callable must also be hashable.
        notify : bool, optional
            Whether to notify for changes. Default is to notify.

        Returns
        -------
        new_expression : ObserverExpression
        """
        return self.then(match(filter=filter, notify=notify))

    def anytrait(self, notify=True):
        """ Create a new expression for observing all traits.

        Events emitted (if any) will be instances of
        :class:`~traits.observation.events.TraitChangeEvent`.

        Parameters
        ----------
        notify : bool, optional
            Whether to notify for changes. Default is to notify.

        Returns
        -------
        new_expression : ObserverExpression
        """
        return self.match(filter=anytrait_filter, notify=notify)

    def metadata(self, metadata_name, notify=True):
        """ Return a new expression for observing traits where the given
        metadata is not None.

        Events emitted (if any) will be instances of
        :class:`~traits.observation.events.TraitChangeEvent`.

        e.g. ``metadata("age")`` matches traits whose 'age' attribute has a
        non-None value.

        Parameters
        ----------
        metadata_name : str
            Name of the metadata to filter traits with.
        notify : bool, optional
            Whether to notify for changes. Default is to notify.

        Returns
        -------
        new_expression : ObserverExpression
        """
        return self.match(filter=MetadataFilter(metadata_name=metadata_name), notify=notify)

    def dict_items(self, notify=True, optional=False):
        """ Create a new expression for observing items inside a dict.

        Events emitted (if any) will be instances of
        :class:`~traits.observation.events.DictChangeEvent`.

        If an expression with ``dict_items`` is further extended, the
        **values** of the dict will be given to the next item in the
        expression. For example, the following observes a trait named "number"
        on any object that is one of the values in the dict named "mapping"::

            trait("mapping").dict_items().trait("number")

        Parameters
        ----------
        notify : bool, optional
            Whether to notify for changes. Default is to notify.
        optional : bool, optional
            Whether to ignore this if the upstream object is not a dict.
            Default is false and an error will be raised if the object is not
            a dict.

        Returns
        -------
        new_expression : ObserverExpression
        """
        return self.then(dict_items(notify=notify, optional=optional))

    def list_items(self, notify=True, optional=False):
        """ Create a new expression for observing items inside a list.

        Events emitted (if any) will be instances of
        :class:`~traits.observation.events.ListChangeEvent`.

        e.g. ``trait("containers").list_items()`` for observing mutations
        to a list named ``containers``.

        e.g. ``trait("containers").list_items().trait("value")`` for observing
        the trait ``value`` on any items in the list ``containers``.

        Parameters
        ----------
        notify : bool, optional
            Whether to notify for changes. Default is to notify.
        optional : bool, optional
            Whether to ignore this if the upstream object is not a list.
            Default is false and an error will be raised if the object is not
            a list.

        Returns
        -------
        new_expression : ObserverExpression
        """
        return self.then(list_items(notify=notify, optional=optional))

    def set_items(self, notify=True, optional=False):
        """ Create a new expression for observing items inside a set.

        Events emitted (if any) will be instances of
        :class:`~traits.observation.events.SetChangeEvent`.

        Parameters
        ----------
        notify : bool, optional
            Whether to notify for changes. Default is to notify.
        optional : bool, optional
            Whether to ignore this if the upstream object is not a set.
            Default is false and an error will be raised if the object is not
            a set.

        Returns
        -------
        new_expression : ObserverExpression
        """
        return self.then(set_items(notify=notify, optional=optional))

    def trait(self, name, notify=True, optional=False):
        """ Create a new expression for observing a trait with the exact
        name given.

        Events emitted (if any) will be instances of
        :class:`~traits.observation.events.TraitChangeEvent`.

        Parameters
        ----------
        name : str
            Name of the trait to match.
        notify : bool, optional
            Whether to notify for changes. Default is to notify.
        optional : bool, optional
            If true, skip this observer if the requested trait is not found.
            Default is false, and an error will be raised if the requested
            trait is not found.

        Returns
        -------
        new_expression : ObserverExpression
        """
        return self.then(trait(name=name, notify=notify, optional=optional))

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

    def _create_graphs(self, branches):
        """ Return a list of ObserverGraph with the given branches.

        Parameters
        ----------
        branches : list of ObserverGraph
            Graphs to be used as branches.

        Returns
        -------
        graphs : list of ObserverGraph
        """
        raise NotImplementedError("'_create_graphs' must be implemented.")