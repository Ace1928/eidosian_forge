import copy
import pickle
import unittest
from traits.has_traits import (
from traits.ctrait import CTrait
from traits.observation.api import (
from traits.observation.exception_handling import (
from traits.traits import ForwardProperty, generic_trait
from traits.trait_types import Event, Float, Instance, Int, List, Map, Str
from traits.trait_errors import TraitError
class TestCallNotifiers(unittest.TestCase):

    def test_trait_and_object_notifiers_called(self):
        side_effects = []

        class Foo(HasTraits):
            x = Int()
            y = Int()

            def _x_changed(self):
                side_effects.append('x')

        def object_handler():
            side_effects.append('object')
        foo = Foo()
        foo.on_trait_change(object_handler, name='anytrait')
        side_effects.clear()
        foo.x = 3
        self.assertEqual(side_effects, ['x', 'object'])
        side_effects.clear()
        foo.y = 4
        self.assertEqual(side_effects, ['object'])

    def test_trait_notifier_modify_object_notifier(self):
        side_effects = []

        def object_handler1():
            side_effects.append('object1')

        def object_handler2():
            side_effects.append('object2')

        class Foo(HasTraits):
            x = Int()
            y = Int()

            def _x_changed(self):
                side_effects.append('x')
                self.on_trait_change(object_handler2, name='anytrait')
        foo = Foo()
        foo.on_trait_change(object_handler1, name='anytrait')
        side_effects.clear()
        foo.x = 1
        self.assertEqual(side_effects, ['x', 'object1'])
        side_effects.clear()
        foo.y = 2
        self.assertEqual(side_effects, ['object1', 'object2'])