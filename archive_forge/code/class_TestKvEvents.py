import unittest
import textwrap
from collections import defaultdict
class TestKvEvents(unittest.TestCase):

    def test_pure_python_auto_binding(self):

        class TestEventsPureAuto(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
        widget = TestEventsPureAuto()
        widget.root_widget = None
        widget.base_widget = widget
        TestEventsPureAuto.check(self)

    def test_pure_python_callbacks(self):

        class TestEventsPure(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
            events_in_pre = [1, 2]
            events_in_applied = [1, 2]
            events_in_post = [1, 2]

            def __init__(self, **kwargs):
                self.fbind('on_kv_pre', lambda _: self.add(2, 'pre'))
                self.fbind('on_kv_applied', lambda _, x: self.add(2, 'applied'))
                self.fbind('on_kv_post', lambda _, x: self.add(2, 'post'))
                super(TestEventsPure, self).__init__(**kwargs)
        widget = TestEventsPure()
        widget.root_widget = None
        widget.base_widget = widget
        widget.fbind('on_kv_pre', lambda _: widget.add(3, 'pre'))
        widget.fbind('on_kv_applied', lambda _, x: widget.add(3, 'applied'))
        widget.fbind('on_kv_post', lambda _, x: widget.add(3, 'post'))
        TestEventsPure.check(self)

    def test_instantiate_from_kv(self):
        from kivy.lang import Builder

        class TestEventsFromKV(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
        widget = Builder.load_string('TestEventsFromKV')
        self.assertIsInstance(widget, TestEventsFromKV)
        widget.root_widget = widget
        widget.base_widget = widget
        widget.check(self)

    def test_instantiate_from_kv_with_event(self):
        from kivy.lang import Builder

        class TestEventsFromKVEvent(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
        widget = Builder.load_string(textwrap.dedent("\n        TestEventsFromKVEvent:\n            events_in_post: [1, 2]\n            on_kv_pre: self.add(2, 'pre')\n            on_kv_applied: self.add(2, 'applied')\n            on_kv_post: self.add(2, 'post')\n            root_widget: self\n            base_widget: self\n        "))
        self.assertIsInstance(widget, TestEventsFromKVEvent)
        widget.check(self)

    def test_instantiate_from_kv_with_child(self):
        from kivy.lang import Builder

        class TestEventsFromKVChild(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
        widget = Builder.load_string(textwrap.dedent("\n        TestEventsFromKVChild:\n            events_in_post: [1, 2]\n            on_kv_pre: self.add(2, 'pre')\n            on_kv_applied: self.add(2, 'applied')\n            on_kv_post: self.add(2, 'post')\n            root_widget: self\n            base_widget: self\n            name: 'root'\n            my_roots_expected_ids: {'child_widget': child_widget}\n            TestEventsFromKVChild:\n                events_in_post: [1, 2]\n                on_kv_pre: self.add(2, 'pre')\n                on_kv_applied: self.add(2, 'applied')\n                on_kv_post: self.add(2, 'post')\n                root_widget: root\n                base_widget: root\n                name: 'child'\n                id: child_widget\n                my_roots_expected_ids: {'child_widget': self}\n        "))
        self.assertIsInstance(widget, TestEventsFromKVChild)
        widget.check(self)

    def test_instantiate_from_kv_with_child_inherit(self):
        from kivy.lang import Builder

        class TestEventsFromKVChildInherit(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
        widget = Builder.load_string(textwrap.dedent("\n        <TestEventsFromKVChildInherit2@TestEventsFromKVChildInherit>:\n            on_kv_pre: self.add(3, 'pre')\n            on_kv_applied: self.add(3, 'applied')\n            on_kv_post: self.add(3, 'post')\n\n        <TestEventsFromKVChildInherit3@TestEventsFromKVChildInherit2>:\n            on_kv_pre: self.add(4, 'pre')\n            on_kv_applied: self.add(4, 'applied')\n            on_kv_post: self.add(4, 'post')\n            some_value: 'fruit'\n            TestEventsFromKVChildInherit2:\n                events_in_applied: [1, 2, 3]\n                events_in_post: [1, 2, 3, 4]\n                on_kv_pre: self.add(4, 'pre')\n                on_kv_applied: self.add(4, 'applied')\n                on_kv_post: self.add(4, 'post')\n                root_widget: root\n                base_widget: self.parent.parent\n                name: 'third child'\n                id: third_child\n                my_roots_expected_ids: {'third_child': self}\n\n        <TestEventsFromKVChildInherit>:\n            on_kv_pre: self.add(2, 'pre')\n            on_kv_applied: self.add(2, 'applied')\n            on_kv_post: self.add(2, 'post')\n            another_value: 'apple'\n\n        TestEventsFromKVChildInherit:\n            events_in_applied: [1, 2]\n            events_in_post: [1, 2, 3]\n            on_kv_pre: self.add(3, 'pre')\n            on_kv_applied: self.add(3, 'applied')\n            on_kv_post: self.add(3, 'post')\n            root_widget: self\n            base_widget: self\n            name: 'root'\n            my_roots_expected_ids:                 {'second_child': second_child, 'first_child': first_child}\n            TestEventsFromKVChildInherit:\n                events_in_applied: [1, 2]\n                events_in_post: [1, 2, 3]\n                on_kv_pre: self.add(3, 'pre')\n                on_kv_applied: self.add(3, 'applied')\n                on_kv_post: self.add(3, 'post')\n                root_widget: root\n                base_widget: root\n                name: 'first child'\n                id: first_child\n                my_roots_expected_ids:                     {'second_child': second_child, 'first_child': self}\n            TestEventsFromKVChildInherit3:\n                events_in_applied: [1, 2, 3, 4]\n                events_in_post: [1, 2, 3, 4, 5]\n                on_kv_pre: self.add(5, 'pre')\n                on_kv_applied: self.add(5, 'applied')\n                on_kv_post: self.add(5, 'post')\n                root_widget: root\n                base_widget: root\n                name: 'second child'\n                some_value: first_child.another_value\n                expected_prop_values: {'some_value': 'apple'}\n                id: second_child\n                my_roots_expected_ids:                     {'second_child': self, 'first_child': first_child}\n        "))
        widget.check(self)