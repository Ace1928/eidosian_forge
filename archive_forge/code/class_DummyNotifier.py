from unittest import mock
from traits.observation._observe import add_or_remove_notifiers
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
class DummyNotifier:
    """ A dummy implementation of INotifier for testing purposes."""

    def add_to(self, observable):
        observable._notifiers(True).append(self)

    def remove_from(self, observable):
        notifiers = observable._notifiers(True)
        try:
            notifiers.remove(self)
        except ValueError:
            raise NotifierNotFound('Notifier not found.')