def _add_or_remove_notifiers(self):
    """ Add or remove user notifiers for the objects observed by the root
        observer.
        """
    if not self.graph.node.notify:
        return
    for observable in self.graph.node.iter_observables(self.object):
        notifier = self.graph.node.get_notifier(handler=self.handler, target=self.target, dispatcher=self.dispatcher)
        if self.remove:
            notifier.remove_from(observable)
        else:
            notifier.add_to(observable)
        self._processed.append((notifier, observable))