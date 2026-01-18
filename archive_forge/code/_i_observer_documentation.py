import abc
 Yield additional ObserverGraph for adding/removing notifiers when
        this observer is encountered in a given ObserverGraph.

        If an observer needs support from another observer(s), e.g.
        for observing 'trait_added' event, then this method can yield any
        number of ObserverGraph containing those additional observer(s).

        If an observer does not need additional support from other observers,
        this method can yield nothing.

        Parameters
        ----------
        graph : ObserverGraph
            The graph where this observer is the root node.

        Yields
        ------
        graph : ObserverGraph
        