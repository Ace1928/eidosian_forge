import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class MovedFunctionTest(test_base.TestCase):

    def test_basics(self):
        self.assertTrue(yellowish_sun())
        self.assertTrue(yellow_sun())
        self.assertEqual('Yellow.', yellowish_sun.__doc__)

    def test_warnings_emitted(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertTrue(yellowish_sun())
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)