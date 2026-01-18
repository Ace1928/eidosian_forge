import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
class DynamicNotifiers(HasTraits):
    ok = Float
    fail = Float
    priority_test = Event
    rebind_calls_0 = List
    rebind_calls_1 = List
    rebind_calls_2 = List
    rebind_calls_3 = List
    rebind_calls_4 = List
    exceptions_from = List
    prioritized_notifications = List

    @on_trait_change('ok')
    def method_listener_0(self):
        self.rebind_calls_0.append(True)

    @on_trait_change('ok')
    def method_listener_1(self, new):
        self.rebind_calls_1.append(new)

    @on_trait_change('ok')
    def method_listener_2(self, name, new):
        self.rebind_calls_2.append((name, new))

    @on_trait_change('ok')
    def method_listener_3(self, obj, name, new):
        self.rebind_calls_3.append((obj, name, new))

    @on_trait_change('ok')
    def method_listener_4(self, obj, name, old, new):
        self.rebind_calls_4.append((obj, name, old, new))

    @on_trait_change('fail')
    def failing_method_listener_0(self):
        self.exceptions_from.append(0)
        raise Exception('error')

    @on_trait_change('fail')
    def failing_method_listener_1(self, new):
        self.exceptions_from.append(1)
        raise Exception('error')

    @on_trait_change('fail')
    def failing_method_listener_2(self, name, new):
        self.exceptions_from.append(2)
        raise Exception('error')

    @on_trait_change('fail')
    def failing_method_listener_3(self, obj, name, new):
        self.exceptions_from.append(3)
        raise Exception('error')

    @on_trait_change('fail')
    def failing_method_listener_4(self, obj, name, old, new):
        self.exceptions_from.append(4)
        raise Exception('error')

    def low_priority_first(self):
        self.prioritized_notifications.append(0)

    def high_priority_first(self):
        self.prioritized_notifications.append(1)

    def low_priority_second(self):
        self.prioritized_notifications.append(2)

    def high_priority_second(self):
        self.prioritized_notifications.append(3)