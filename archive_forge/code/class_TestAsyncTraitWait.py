import random
import threading
import time
import unittest
from traits.api import Enum, HasStrictTraits
from traits.util.async_trait_wait import wait_for_condition
class TestAsyncTraitWait(unittest.TestCase):

    def test_wait_for_condition_success(self):
        lights = TrafficLights(colour='Green')
        t = threading.Thread(target=lights.make_random_changes, args=(2,))
        t.start()
        wait_for_condition(condition=lambda l: l.colour == 'Red', obj=lights, trait='colour')
        self.assertEqual(lights.colour, 'Red')
        t.join()

    def test_wait_for_condition_failure(self):
        lights = TrafficLights(colour='Green')
        t = threading.Thread(target=lights.make_random_changes, args=(2,))
        t.start()
        self.assertRaises(RuntimeError, wait_for_condition, condition=lambda l: l.colour == 'RedAndAmber', obj=lights, trait='colour', timeout=5.0)
        t.join()

    def test_traits_handler_cleaned_up(self):
        self.lights = TrafficLights(colour='Green')
        t = threading.Thread(target=self.lights.make_random_changes, args=(3,))
        t.start()
        wait_for_condition(condition=lambda l: self.lights.colour == 'Red', obj=self.lights, trait='colour')
        del self.lights
        t.join()