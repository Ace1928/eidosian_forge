import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
class TestObserverChangeNotifierWeakrefTarget(unittest.TestCase):
    """ Tests for weak references on targets.
    """

    def test_target_can_be_garbage_collected(self):
        target = mock.Mock()
        target_ref = weakref.ref(target)
        notifier = create_notifier(target=target)
        del target
        self.assertIsNone(target_ref())

    def test_deleted_target_silence_notifier(self):
        target = mock.Mock()
        observer_handler = mock.Mock()
        notifier = create_notifier(observer_handler=observer_handler, target=target)
        del target
        notifier(a=1, b=2)
        observer_handler.assert_not_called()