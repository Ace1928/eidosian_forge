import pyomo.common.unittest as unittest
from pyomo.common.tee import capture_output
import gc
from io import StringIO
from itertools import zip_longest
import logging
import sys
import time
from pyomo.common.log import LoggingIntercept
from pyomo.common.timing import (
from pyomo.environ import (
from pyomo.core.base.var import _VarData
class TestFlattenHierarchicalTimer(unittest.TestCase):

    def make_singleton_timer(self):
        timer = HierarchicalTimer()
        timer.start('root')
        timer.stop('root')
        timer.timers['root'].total_time = 5.0
        return timer

    def make_flat_timer(self):
        timer = HierarchicalTimer()
        timer.start('root')
        timer.start('a')
        timer.stop('a')
        timer.start('b')
        timer.stop('b')
        timer.stop('root')
        timer.timers['root'].total_time = 5.0
        timer.timers['root'].timers['a'].total_time = 1.0
        timer.timers['root'].timers['b'].total_time = 2.5
        return timer

    def make_timer_depth_2_one_child(self):
        timer = HierarchicalTimer()
        timer.start('root')
        timer.start('a')
        timer.start('b')
        timer.stop('b')
        timer.start('c')
        timer.stop('c')
        timer.stop('a')
        timer.stop('root')
        timer.timers['root'].total_time = 5.0
        timer.timers['root'].timers['a'].total_time = 4.0
        timer.timers['root'].timers['a'].timers['b'].total_time = 1.1
        timer.timers['root'].timers['a'].timers['c'].total_time = 2.2
        return timer

    def make_timer_depth_2_with_name_collision(self):
        timer = HierarchicalTimer()
        timer.start('root')
        timer.start('a')
        timer.start('b')
        timer.stop('b')
        timer.start('c')
        timer.stop('c')
        timer.stop('a')
        timer.start('b')
        timer.stop('b')
        timer.stop('root')
        timer.timers['root'].total_time = 5.0
        timer.timers['root'].timers['a'].total_time = 4.0
        timer.timers['root'].timers['a'].timers['b'].total_time = 1.1
        timer.timers['root'].timers['a'].timers['c'].total_time = 2.2
        timer.timers['root'].timers['b'].total_time = 0.11
        return timer

    def make_timer_depth_2_two_children(self):
        timer = HierarchicalTimer()
        timer.start('root')
        timer.start('a')
        timer.start('b')
        timer.stop('b')
        timer.start('c')
        timer.stop('c')
        timer.stop('a')
        timer.start('b')
        timer.start('c')
        timer.stop('c')
        timer.start('d')
        timer.stop('d')
        timer.stop('b')
        timer.stop('root')
        timer.timers['root'].total_time = 5.0
        timer.timers['root'].timers['a'].total_time = 4.0
        timer.timers['root'].timers['a'].timers['b'].total_time = 1.1
        timer.timers['root'].timers['a'].timers['c'].total_time = 2.2
        timer.timers['root'].timers['b'].total_time = 0.88
        timer.timers['root'].timers['b'].timers['c'].total_time = 0.07
        timer.timers['root'].timers['b'].timers['d'].total_time = 0.05
        return timer

    def make_timer_depth_4(self):
        timer = HierarchicalTimer()
        timer.start('root')
        timer.start('a')
        timer.start('b')
        timer.stop('b')
        timer.start('c')
        timer.start('d')
        timer.start('e')
        timer.stop('e')
        timer.stop('d')
        timer.stop('c')
        timer.stop('a')
        timer.start('b')
        timer.start('c')
        timer.start('e')
        timer.stop('e')
        timer.stop('c')
        timer.start('d')
        timer.stop('d')
        timer.stop('b')
        timer.stop('root')
        timer.timers['root'].total_time = 5.0
        timer.timers['root'].timers['a'].total_time = 4.0
        timer.timers['root'].timers['a'].timers['b'].total_time = 1.1
        timer.timers['root'].timers['a'].timers['c'].total_time = 2.2
        timer.timers['root'].timers['a'].timers['c'].timers['d'].total_time = 0.9
        timer.timers['root'].timers['a'].timers['c'].timers['d'].timers['e'].total_time = 0.6
        timer.timers['root'].timers['b'].total_time = 0.88
        timer.timers['root'].timers['b'].timers['c'].total_time = 0.07
        timer.timers['root'].timers['b'].timers['c'].timers['e'].total_time = 0.04
        timer.timers['root'].timers['b'].timers['d'].total_time = 0.05
        return timer

    def make_timer_depth_4_same_name(self):
        timer = HierarchicalTimer()
        timer.start('root')
        timer.start('a')
        timer.start('a')
        timer.start('a')
        timer.start('a')
        timer.stop('a')
        timer.stop('a')
        timer.stop('a')
        timer.stop('a')
        timer.stop('root')
        timer.timers['root'].total_time = 5.0
        timer.timers['root'].timers['a'].total_time = 1.0
        timer.timers['root'].timers['a'].timers['a'].total_time = 0.1
        timer.timers['root'].timers['a'].timers['a'].timers['a'].total_time = 0.01
        timer.timers['root'].timers['a'].timers['a'].timers['a'].timers['a'].total_time = 0.001
        return timer

    def test_singleton(self):
        timer = self.make_singleton_timer()
        root = timer.timers['root']
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)

    def test_already_flat(self):
        timer = self.make_flat_timer()
        root = timer.timers['root']
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers['a'].total_time, 1.0)
        self.assertAlmostEqual(root.timers['b'].total_time, 2.5)

    def test_depth_2_one_child(self):
        timer = self.make_timer_depth_2_one_child()
        root = timer.timers['root']
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers['a'].total_time, 0.7)
        self.assertAlmostEqual(root.timers['b'].total_time, 1.1)
        self.assertAlmostEqual(root.timers['c'].total_time, 2.2)

    def test_timer_depth_2_with_name_collision(self):
        timer = self.make_timer_depth_2_with_name_collision()
        root = timer.timers['root']
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers['a'].total_time, 0.7)
        self.assertAlmostEqual(root.timers['b'].total_time, 1.21)
        self.assertAlmostEqual(root.timers['c'].total_time, 2.2)

    def test_timer_depth_2_two_children(self):
        timer = self.make_timer_depth_2_two_children()
        root = timer.timers['root']
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers['a'].total_time, 0.7)
        self.assertAlmostEqual(root.timers['b'].total_time, 1.86)
        self.assertAlmostEqual(root.timers['c'].total_time, 2.27)
        self.assertAlmostEqual(root.timers['d'].total_time, 0.05)

    def test_timer_depth_4(self):
        timer = self.make_timer_depth_4()
        root = timer.timers['root']
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers['a'].total_time, 0.7)
        self.assertAlmostEqual(root.timers['b'].total_time, 1.86)
        self.assertAlmostEqual(root.timers['c'].total_time, 1.33)
        self.assertAlmostEqual(root.timers['d'].total_time, 0.35)
        self.assertAlmostEqual(root.timers['e'].total_time, 0.64)

    def test_timer_depth_4_same_name(self):
        timer = self.make_timer_depth_4_same_name()
        root = timer.timers['root']
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers['a'].total_time, 1.0)

    def test_base_timer_depth_3(self):
        timer = self.make_timer_depth_2_two_children()
        timer.flatten()
        self.assertAlmostEqual(timer.timers['root'].total_time, 0.12)
        self.assertAlmostEqual(timer.timers['a'].total_time, 0.7)
        self.assertAlmostEqual(timer.timers['b'].total_time, 1.86)
        self.assertAlmostEqual(timer.timers['c'].total_time, 2.27)
        self.assertAlmostEqual(timer.timers['d'].total_time, 0.05)

    def test_timer_still_active(self):
        timer = HierarchicalTimer()
        timer.start('a')
        timer.stop('a')
        timer.start('b')
        msg = 'Cannot flatten.*while any timers are active'
        with self.assertRaisesRegex(RuntimeError, msg):
            timer.flatten()
        timer.stop('b')