import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class MovedInheritableClassTest(test_base.TestCase):

    def test_broken_type_class(self):
        self.assertRaises(TypeError, moves.moved_class, 'b', __name__)

    def test_basics(self):
        old = OldHotness()
        self.assertIsInstance(old, NewHotness)
        self.assertEqual('cold', old.hot())

    def test_warnings_emitted_creation(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            OldHotness()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_warnings_emitted_creation_pending(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            OldHotness2()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)

    def test_existing_refer_subclass(self):

        class MyOldThing(OldHotness):
            pass
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            MyOldThing()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)