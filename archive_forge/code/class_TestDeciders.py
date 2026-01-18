from taskflow import deciders
from taskflow import test
class TestDeciders(test.TestCase):

    def test_translate(self):
        for val in ['all', 'ALL', 'aLL', deciders.Depth.ALL]:
            self.assertEqual(deciders.Depth.ALL, deciders.Depth.translate(val))
        for val in ['atom', 'ATOM', 'atOM', deciders.Depth.ATOM]:
            self.assertEqual(deciders.Depth.ATOM, deciders.Depth.translate(val))
        for val in ['neighbors', 'Neighbors', 'NEIGHBORS', deciders.Depth.NEIGHBORS]:
            self.assertEqual(deciders.Depth.NEIGHBORS, deciders.Depth.translate(val))
        for val in ['flow', 'FLOW', 'flOW', deciders.Depth.FLOW]:
            self.assertEqual(deciders.Depth.FLOW, deciders.Depth.translate(val))

    def test_bad_translate(self):
        self.assertRaises(TypeError, deciders.Depth.translate, 3)
        self.assertRaises(TypeError, deciders.Depth.translate, object())
        self.assertRaises(ValueError, deciders.Depth.translate, 'stuff')

    def test_pick_widest(self):
        choices = [deciders.Depth.ATOM, deciders.Depth.FLOW]
        self.assertEqual(deciders.Depth.FLOW, deciders.pick_widest(choices))
        choices = [deciders.Depth.ATOM, deciders.Depth.FLOW, deciders.Depth.ALL]
        self.assertEqual(deciders.Depth.ALL, deciders.pick_widest(choices))
        choices = [deciders.Depth.ATOM, deciders.Depth.FLOW, deciders.Depth.ALL, deciders.Depth.NEIGHBORS]
        self.assertEqual(deciders.Depth.ALL, deciders.pick_widest(choices))
        choices = [deciders.Depth.ATOM, deciders.Depth.NEIGHBORS]
        self.assertEqual(deciders.Depth.NEIGHBORS, deciders.pick_widest(choices))

    def test_bad_pick_widest(self):
        self.assertRaises(ValueError, deciders.pick_widest, [])
        self.assertRaises(ValueError, deciders.pick_widest, ['a'])
        self.assertRaises(ValueError, deciders.pick_widest, set(['b']))