import unittest
from traits.api import (
from traits.observation.api import (
class TestHasTraitsObserverDifferentiateParent(unittest.TestCase):

    def test_shared_instance_but_different_target(self):
        potato = Potato()
        potato_bag = PotatoBag(potatos=[potato])
        crate1 = Crate(potato_bags=[potato_bag])
        crate2 = Crate(potato_bags=[potato_bag])
        events = []
        handler = events.append
        crate1.observe(handler, 'potato_bags:items:potatos:items:name')
        crate2.observe(handler, 'potato_bags:items:potatos:items:name')
        potato.name = 'King Edward'
        self.assertEqual(len(events), 2)

    def test_shared_instance_same_graph_different_target(self):
        crate1 = Crate()
        crate2 = Crate()
        events = []
        handler = events.append
        crate1.observe(handler, 'potato_bags:items:potatos:items:name')
        crate2.observe(handler, 'potato_bags:items:potatos:items:name')
        new_potato = Potato()
        new_potato_bag = PotatoBag(potatos=[new_potato])
        crate1.potato_bags = [new_potato_bag]
        crate2.potato_bags = [new_potato_bag]
        new_potato.name = 'King Edward I'
        self.assertEqual(len(events), 2)
        events.clear()
        crate2.observe(handler, 'potato_bags:items:potatos:items:name', remove=True)
        new_potato.name = 'King Edward II'
        self.assertEqual(len(events), 1)
        events.clear()
        maris_piper = Potato()
        crate2.potato_bags[0].potatos.append(maris_piper)
        crate1.potato_bags = []
        self.assertEqual(len(events), 0)
        maris_piper.name = 'Maris Piper'
        self.assertEqual(len(events), 0)