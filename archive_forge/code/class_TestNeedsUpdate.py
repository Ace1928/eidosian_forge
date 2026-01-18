from heat.tests import common
from heat.scaling import rolling_update
class TestNeedsUpdate(common.HeatTestCase):
    scenarios = [('4_4_0', dict(targ=4, curr=4, updated=0, result=True)), ('4_4_1', dict(targ=4, curr=4, updated=1, result=True)), ('4_4_3', dict(targ=4, curr=4, updated=3, result=True)), ('4_4_4', dict(targ=4, curr=4, updated=4, result=False)), ('4_4_5', dict(targ=4, curr=4, updated=5, result=False)), ('4_5_0', dict(targ=4, curr=5, updated=0, result=True)), ('4_5_1', dict(targ=4, curr=5, updated=1, result=True)), ('4_5_3', dict(targ=4, curr=5, updated=3, result=True)), ('4_5_4', dict(targ=4, curr=5, updated=4, result=True)), ('4_5_5', dict(targ=4, curr=5, updated=5, result=True)), ('4_3_0', dict(targ=4, curr=3, updated=0, result=True)), ('4_3_1', dict(targ=4, curr=3, updated=1, result=True)), ('4_3_2', dict(targ=4, curr=3, updated=2, result=True)), ('4_3_3', dict(targ=4, curr=3, updated=3, result=True)), ('4_3_4', dict(targ=4, curr=3, updated=4, result=True))]

    def test_needs_update(self):
        needs_update = rolling_update.needs_update(self.targ, self.curr, self.updated)
        self.assertEqual(self.result, needs_update)