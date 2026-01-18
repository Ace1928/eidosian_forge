import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class MovedPropertyTest(test_base.TestCase):

    def test_basics(self):
        dog = WoofWoof()
        self.assertEqual('woof', dog.burk)
        self.assertEqual('woof', dog.bark)

    def test_readonly_move(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual('orange', Giraffe.colour)
            g = Giraffe()
            self.assertEqual(2, g.heightt)
        self.assertEqual(2, len(capture))

    def test_warnings_emitted(self):
        dog = WoofWoof()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual('woof', dog.burk)
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_warnings_emitted_pending(self):
        dog = WoofWoof()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual('woof', dog.berk)
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)

    def test_warnings_not_emitted(self):
        dog = WoofWoof()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual('woof', dog.bark)
        self.assertEqual(0, len(capture))