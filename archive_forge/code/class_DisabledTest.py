import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class DisabledTest(test_base.TestCase):

    def test_basics(self):
        dog = WoofWoof()
        c = KittyKat()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            with disable.DisableFixture():
                self.assertTrue(yellowish_sun())
                self.assertEqual('woof', dog.berk)
                self.assertEqual('supermeow', c.meow())
        self.assertEqual(0, len(capture))