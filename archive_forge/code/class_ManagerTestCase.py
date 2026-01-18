from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
class ManagerTestCase(base.ServiceBaseTestCase):
    """Tests the periodic tasks portion of the manager class."""

    def setUp(self):
        super(ManagerTestCase, self).setUp()

    def test_periodic_tasks_with_idle(self):

        class Manager(periodic_task.PeriodicTasks):

            @periodic_task.periodic_task(spacing=200)
            def bar(self):
                return 'bar'
        m = Manager(self.conf)
        self.assertThat(m._periodic_tasks, matchers.HasLength(1))
        self.assertEqual(200, m._periodic_spacing['bar'])
        idle = m.run_periodic_tasks(None)
        self.assertAlmostEqual(60, idle, 1)

    def test_periodic_tasks_constant(self):

        class Manager(periodic_task.PeriodicTasks):

            @periodic_task.periodic_task(spacing=0)
            def bar(self):
                return 'bar'
        m = Manager(self.conf)
        idle = m.run_periodic_tasks(None)
        self.assertAlmostEqual(60, idle, 1)

    @mock.patch('oslo_service.periodic_task.now')
    def test_periodic_tasks_idle_calculation(self, mock_now):
        fake_time = 32503680000.0
        mock_now.return_value = fake_time

        class Manager(periodic_task.PeriodicTasks):

            @periodic_task.periodic_task(spacing=10)
            def bar(self, context):
                return 'bar'
        m = Manager(self.conf)
        self.assertEqual(1, len(m._periodic_tasks))
        task_name, task = m._periodic_tasks[0]
        self.assertEqual('bar', task_name)
        self.assertEqual(10, task._periodic_spacing)
        self.assertTrue(task._periodic_enabled)
        self.assertFalse(task._periodic_external_ok)
        self.assertFalse(task._periodic_immediate)
        self.assertAlmostEqual(32503680000.0, task._periodic_last_run)
        self.assertEqual(10, m._periodic_spacing[task_name])
        self.assertAlmostEqual(32503680000.0, m._periodic_last_run[task_name])
        mock_now.return_value = fake_time + 5
        idle = m.run_periodic_tasks(None)
        self.assertAlmostEqual(5, idle, 1)
        self.assertAlmostEqual(32503680000.0, m._periodic_last_run[task_name])
        mock_now.return_value = fake_time + 10
        idle = m.run_periodic_tasks(None)
        self.assertAlmostEqual(10, idle, 1)
        self.assertAlmostEqual(32503680010.0, m._periodic_last_run[task_name])

    @mock.patch('oslo_service.periodic_task.now')
    def test_periodic_tasks_immediate_runs_now(self, mock_now):
        fake_time = 32503680000.0
        mock_now.return_value = fake_time

        class Manager(periodic_task.PeriodicTasks):

            @periodic_task.periodic_task(spacing=10, run_immediately=True)
            def bar(self, context):
                return 'bar'
        m = Manager(self.conf)
        self.assertEqual(1, len(m._periodic_tasks))
        task_name, task = m._periodic_tasks[0]
        self.assertEqual('bar', task_name)
        self.assertEqual(10, task._periodic_spacing)
        self.assertTrue(task._periodic_enabled)
        self.assertFalse(task._periodic_external_ok)
        self.assertTrue(task._periodic_immediate)
        self.assertIsNone(task._periodic_last_run)
        self.assertEqual(10, m._periodic_spacing[task_name])
        self.assertIsNone(m._periodic_last_run[task_name])
        idle = m.run_periodic_tasks(None)
        self.assertAlmostEqual(32503680000.0, m._periodic_last_run[task_name])
        self.assertAlmostEqual(10, idle, 1)
        mock_now.return_value = fake_time + 5
        idle = m.run_periodic_tasks(None)
        self.assertAlmostEqual(5, idle, 1)

    def test_periodic_tasks_disabled(self):

        class Manager(periodic_task.PeriodicTasks):

            @periodic_task.periodic_task(spacing=-1)
            def bar(self):
                return 'bar'
        m = Manager(self.conf)
        idle = m.run_periodic_tasks(None)
        self.assertAlmostEqual(60, idle, 1)

    def test_external_running_here(self):
        self.config(run_external_periodic_tasks=True)

        class Manager(periodic_task.PeriodicTasks):

            @periodic_task.periodic_task(spacing=200, external_process_ok=True)
            def bar(self):
                return 'bar'
        m = Manager(self.conf)
        self.assertThat(m._periodic_tasks, matchers.HasLength(1))

    @mock.patch('oslo_service.periodic_task.now')
    @mock.patch('random.random')
    def test_nearest_boundary(self, mock_random, mock_now):
        mock_now.return_value = 19
        mock_random.return_value = 0
        self.assertEqual(17, periodic_task._nearest_boundary(10, 7))
        mock_now.return_value = 28
        self.assertEqual(27, periodic_task._nearest_boundary(13, 7))
        mock_now.return_value = 1841
        self.assertEqual(1837, periodic_task._nearest_boundary(781, 88))
        mock_now.return_value = 1835
        self.assertEqual(mock_now.return_value, periodic_task._nearest_boundary(None, 88))
        mock_random.return_value = 1.0
        mock_now.return_value = 1300
        self.assertEqual(1200 + 10, periodic_task._nearest_boundary(1000, 200))
        mock_random.return_value = 0.5
        mock_now.return_value = 1300
        self.assertEqual(1200 + 5, periodic_task._nearest_boundary(1000, 200))