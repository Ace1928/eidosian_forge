import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class RemovalTests(test_base.TestCase):

    def test_function_args(self):
        self.assertEqual(666, crimson_lightning(666))

    def test_function_noargs(self):
        self.assertTrue(red_comet())

    def test_function_keeps_argspec(self):
        self.assertEqual(inspect.getfullargspec(crimson_lightning_unwrapped), inspect.getfullargspec(crimson_lightning))

    def test_deprecated_kwarg(self):

        @removals.removed_kwarg('b')
        def f(b=2):
            return b
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual(3, f(b=3))
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual(2, f())
        self.assertEqual(0, len(capture))

    def test_removed_kwarg_keeps_argspec(self):

        @removals.removed_kwarg('b')
        def f(b=2):
            return b

        def f_unwrapped(b=2):
            return b
        self.assertEqual(inspect.getfullargspec(f_unwrapped), inspect.getfullargspec(f))

    def test_pending_deprecated_kwarg(self):

        @removals.removed_kwarg('b', category=PendingDeprecationWarning)
        def f(b=2):
            return b
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual(3, f(b=3))
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual(2, f())
        self.assertEqual(0, len(capture))

    def test_warnings_emitted_property(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            o = ThingB()
            self.assertEqual('green', o.green_tristars)
            o.green_tristars = 'b'
            del o.green_tristars
        self.assertEqual(3, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_warnings_emitted_property_custom_message(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            o = ThingB()
            self.assertEqual('green-blue', o.green_blue_tristars)
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertIn('stop using me', str(w.message))
        self.assertEqual(DeprecationWarning, w.category)

    def test_warnings_emitted_function_args(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual(666, crimson_lightning(666))
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_pending_warnings_emitted_function_args(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual(666, crimson_lightning_to_remove(666))
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)

    def test_warnings_emitted_function_noargs(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertTrue(red_comet())
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_pending_warnings_emitted_function_noargs(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertTrue(blue_comet())
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)

    def test_warnings_emitted_class(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            EFSF()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_pending_warnings_emitted_class(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            EFSF_2()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)

    def test_pending_warnings_emitted_class_direct(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            s = StarLord()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)
        self.assertEqual('star', s.name)

    def test_pending_warnings_emitted_class_inherit(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            s = StarLordJr('star_jr')
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)
        self.assertEqual('star_jr', s.name)

    def test_warnings_emitted_instancemethod(self):
        zeon = ThingB()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            zeon.black_tristars()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_pending_warnings_emitted_instancemethod(self):
        zeon = ThingB()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            zeon.blue_tristars()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)

    def test_pending_warnings_emitted_classmethod(self):
        zeon = ThingB()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            zeon.yellow_wolf()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)

    def test_warnings_emitted_classmethod(self):
        zeon = ThingB()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            zeon.white_wolf()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_warnings_emitted_staticmethod(self):
        zeon = ThingB()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            zeon.blue_giant()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_pending_warnings_emitted_staticmethod(self):
        zeon = ThingB()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            zeon.green_giant()
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)

    def test_removed_module(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            removals.removed_module(__name__)
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_pending_removed_module(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            removals.removed_module(__name__, category=PendingDeprecationWarning)
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)

    def test_removed_module_bad_type(self):
        self.assertRaises(TypeError, removals.removed_module, 2)