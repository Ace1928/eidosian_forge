import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class RenamedKwargTest(test_base.TestCase):

    def test_basics(self):
        self.assertEqual((1, 1), blip_blop())
        self.assertEqual((2, 1), blip_blop(blip=2))
        self.assertEqual((1, 2), blip_blop(blop=2))
        self.assertEqual((2, 2), blip_blop(blip=2, blop=2))
        self.assertEqual(2, blip_blop_3(blip=2))
        self.assertEqual(2, blip_blop_3(blop=2))

    def test_warnings_emitted(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual((2, 1), blip_blop(blip=2))
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual(2, blip_blop_3(blip=2))
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_warnings_emitted_classmethod(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            WoofWoof.factory(resp='hi')
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            WoofWoof.factory(response='hi')
        self.assertEqual(0, len(capture))

    def test_warnings_emitted_pending(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual((2, 1), blip_blop_2(blip=2))
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)

    def test_warnings_not_emitted(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual((1, 2), blip_blop(blop=2))
        self.assertEqual(0, len(capture))
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual(2, blip_blop_3(blop=2))
        self.assertEqual(0, len(capture))

    def test_argspec(self):
        self.assertEqual(inspect.getfullargspec(blip_blop_unwrapped), inspect.getfullargspec(blip_blop))