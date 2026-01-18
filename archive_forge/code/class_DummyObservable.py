import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
class DummyObservable:
    """ Dummy implementation of IObservable for testing purposes.
    """

    def __init__(self):
        self.notifiers = []

    def _notifiers(self, force_create):
        return self.notifiers

    def handler(self, event):
        pass