from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
class TargetedGraphFlowTest(test.TestCase):

    def test_targeted_flow_restricts(self):
        f = gf.TargetedFlow('test')
        task1 = _task('task1', provides=['a'], requires=[])
        task2 = _task('task2', provides=['b'], requires=['a'])
        task3 = _task('task3', provides=[], requires=['b'])
        task4 = _task('task4', provides=[], requires=['b'])
        f.add(task1, task2, task3, task4)
        f.set_target(task3)
        self.assertEqual(3, len(f))
        self.assertCountEqual(f, [task1, task2, task3])
        self.assertNotIn('c', f.provides)

    def test_targeted_flow_reset(self):
        f = gf.TargetedFlow('test')
        task1 = _task('task1', provides=['a'], requires=[])
        task2 = _task('task2', provides=['b'], requires=['a'])
        task3 = _task('task3', provides=[], requires=['b'])
        task4 = _task('task4', provides=['c'], requires=['b'])
        f.add(task1, task2, task3, task4)
        f.set_target(task3)
        f.reset_target()
        self.assertEqual(4, len(f))
        self.assertCountEqual(f, [task1, task2, task3, task4])
        self.assertIn('c', f.provides)

    def test_targeted_flow_bad_target(self):
        f = gf.TargetedFlow('test')
        task1 = _task('task1', provides=['a'], requires=[])
        task2 = _task('task2', provides=['b'], requires=['a'])
        f.add(task1)
        self.assertRaisesRegex(ValueError, '^Node .* not found', f.set_target, task2)

    def test_targeted_flow_one_node(self):
        f = gf.TargetedFlow('test')
        task1 = _task('task1', provides=['a'], requires=[])
        f.add(task1)
        f.set_target(task1)
        self.assertEqual(1, len(f))
        self.assertCountEqual(f, [task1])

    def test_recache_on_add(self):
        f = gf.TargetedFlow('test')
        task1 = _task('task1', provides=[], requires=['a'])
        f.add(task1)
        f.set_target(task1)
        self.assertEqual(1, len(f))
        task2 = _task('task2', provides=['a'], requires=[])
        f.add(task2)
        self.assertEqual(2, len(f))

    def test_recache_on_add_no_deps(self):
        f = gf.TargetedFlow('test')
        task1 = _task('task1', provides=[], requires=[])
        f.add(task1)
        f.set_target(task1)
        self.assertEqual(1, len(f))
        task2 = _task('task2', provides=[], requires=[])
        f.add(task2)
        self.assertEqual(1, len(f))

    def test_recache_on_link(self):
        f = gf.TargetedFlow('test')
        task1 = _task('task1', provides=[], requires=[])
        task2 = _task('task2', provides=[], requires=[])
        f.add(task1, task2)
        f.set_target(task1)
        self.assertEqual(1, len(f))
        f.link(task2, task1)
        self.assertEqual(2, len(f))
        self.assertEqual([(task2, task1, {'manual': True})], list(f.iter_links()))