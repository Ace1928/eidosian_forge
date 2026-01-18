import unittest
from traits.api import (
from traits.observation.api import (
class TestHasTraitsObserveListOfInstance(unittest.TestCase):

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

    def test_observe_instance_in_nested_list(self):
        container = ClassWithListOfListOfInstance()
        events = []
        handler = events.append
        container.observe(expression=trait('list_of_list_of_instances', notify=False).list_items(notify=False).list_items(notify=False).trait('value'), handler=handler)
        single_value_instance = SingleValue()
        inner_list = [single_value_instance]
        container.list_of_list_of_instances.append(inner_list)
        self.assertEqual(len(events), 0)
        single_value_instance.value += 1
        event, = events
        self.assertEqual(event.object, single_value_instance)
        self.assertEqual(event.name, 'value')
        self.assertEqual(event.old, 0)
        self.assertEqual(event.new, 1)

    def test_nested_list_reassigned_value_compared_equally(self):
        container = ClassWithListOfListOfInstance()
        events = []
        handler = events.append
        container.observe(expression=trait('list_of_list_of_instances', notify=False).list_items(notify=False).list_items(notify=False).trait('value'), handler=handler)
        inner_list = [SingleValue()]
        container.list_of_list_of_instances = [inner_list]
        self.assertEqual(len(events), 0)
        container.list_of_list_of_instances[0] = inner_list
        second_instance = SingleValue()
        container.list_of_list_of_instances[0].append(second_instance)
        self.assertEqual(len(events), 0)
        second_instance.value += 1
        event, = events
        self.assertEqual(event.object, second_instance)
        self.assertEqual(event.name, 'value')
        self.assertEqual(event.old, 0)
        self.assertEqual(event.new, 1)

    def test_duplicated_items_tracked(self):
        container = ClassWithListOfInstance()
        events = []
        handler = events.append
        container.observe(expression=trait('list_of_instances', notify=False).list_items(notify=False).trait('value'), handler=handler)
        instance = SingleValue()
        container.list_of_instances.append(instance)
        container.list_of_instances.append(instance)
        self.assertEqual(len(events), 0)
        instance.value += 1
        self.assertEqual(len(events), 1)
        events.clear()
        container.list_of_instances.pop()
        instance.value += 1
        self.assertEqual(len(events), 1)
        events.clear()
        container.list_of_instances.pop()
        instance.value += 1
        self.assertEqual(len(events), 0)