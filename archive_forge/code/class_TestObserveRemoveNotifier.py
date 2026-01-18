import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_types import Instance, Int
from traits.observation.api import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation.expression import compile_expr, trait
from traits.observation.observe import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
class TestObserveRemoveNotifier(unittest.TestCase):
    """ Test the remove action."""

    def test_remove_trait_notifiers(self):
        observable = DummyObservable()
        notifier = DummyNotifier()
        observable.notifiers = [notifier]
        observer = DummyObserver(observables=[observable], notifier=notifier)
        graph = ObserverGraph(node=observer)
        call_add_or_remove_notifiers(graph=graph, remove=True)
        self.assertEqual(observable.notifiers, [])

    def test_remove_notifiers_skip_if_notify_flag_is_false(self):
        observable = DummyObservable()
        notifier = DummyNotifier()
        observable.notifiers = [notifier]
        observer = DummyObserver(notify=False, observables=[observable], notifier=notifier)
        graph = ObserverGraph(node=observer)
        call_add_or_remove_notifiers(graph=graph, remove=True)
        self.assertEqual(observable.notifiers, [notifier])

    def test_remove_maintainers(self):
        observable = DummyObservable()
        maintainer = DummyNotifier()
        observable.notifiers = [maintainer, maintainer]
        root_observer = DummyObserver(notify=False, observables=[observable], maintainer=maintainer)
        graph = ObserverGraph(node=root_observer, children=[ObserverGraph(node=DummyObserver()), ObserverGraph(node=DummyObserver())])
        call_add_or_remove_notifiers(graph=graph, remove=True)
        self.assertEqual(observable.notifiers, [])

    def test_remove_notifiers_for_children_graphs(self):
        observable1 = DummyObservable()
        notifier1 = DummyNotifier()
        child_observer1 = DummyObserver(observables=[observable1], notifier=notifier1)
        observable2 = DummyObservable()
        notifier2 = DummyNotifier()
        child_observer2 = DummyObserver(observables=[observable2], notifier=notifier2)
        parent_observer = DummyObserver(next_objects=[mock.Mock()])
        graph = ObserverGraph(node=parent_observer, children=[ObserverGraph(node=child_observer1), ObserverGraph(node=child_observer2)])
        observable1.notifiers = [notifier1]
        observable2.notifiers = [notifier2]
        call_add_or_remove_notifiers(graph=graph, remove=True)
        self.assertEqual(observable1.notifiers, [])
        self.assertEqual(observable2.notifiers, [])

    def test_remove_notifiers_for_extra_graph(self):
        observable = DummyObservable()
        extra_notifier = DummyNotifier()
        extra_observer = DummyObserver(observables=[observable], notifier=extra_notifier)
        extra_graph = ObserverGraph(node=extra_observer)
        observer = DummyObserver(extra_graphs=[extra_graph])
        graph = ObserverGraph(node=observer)
        observable.notifiers = [extra_notifier]
        call_add_or_remove_notifiers(graph=graph, remove=True)
        self.assertEqual(observable.notifiers, [])

    def test_remove_notifier_raises_let_error_propagate(self):
        observer = DummyObserver(observables=[DummyObservable()], notifier=DummyNotifier())
        with self.assertRaises(NotifierNotFound):
            call_add_or_remove_notifiers(graph=ObserverGraph(node=observer), remove=True)

    def test_remove_atomic(self):
        notifier = DummyNotifier()
        maintainer = DummyNotifier()
        observable1 = DummyObservable()
        observable1.notifiers = [notifier, maintainer]
        old_observable1_notifiers = observable1.notifiers.copy()
        observable2 = DummyObservable()
        observable2.notifiers = [maintainer]
        old_observable2_notifiers = observable2.notifiers.copy()
        observable3 = DummyObservable()
        observable3.notifiers = [notifier, maintainer]
        old_observable3_notifiers = observable3.notifiers.copy()
        observer = DummyObserver(notify=True, observables=[observable1, observable2, observable3], notifier=notifier, maintainer=maintainer)
        graph = create_graph(observer, DummyObserver())
        with self.assertRaises(NotifierNotFound):
            call_add_or_remove_notifiers(object=mock.Mock(), graph=graph, remove=True)
        self.assertCountEqual(observable1.notifiers, old_observable1_notifiers)
        self.assertCountEqual(observable2.notifiers, old_observable2_notifiers)
        self.assertCountEqual(observable3.notifiers, old_observable3_notifiers)