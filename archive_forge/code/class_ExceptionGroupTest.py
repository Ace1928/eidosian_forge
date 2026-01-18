import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
class ExceptionGroupTest(common.HeatTestCase):

    def test_contains_exceptions(self):
        exception_group = scheduler.ExceptionGroup()
        self.assertIsInstance(exception_group.exceptions, list)

    def test_can_be_initialized_with_a_list_of_exceptions(self):
        ex1 = Exception('ex 1')
        ex2 = Exception('ex 2')
        exception_group = scheduler.ExceptionGroup([ex1, ex2])
        self.assertIn(ex1, exception_group.exceptions)
        self.assertIn(ex2, exception_group.exceptions)

    def test_can_add_exceptions_after_init(self):
        ex = Exception()
        exception_group = scheduler.ExceptionGroup()
        exception_group.exceptions.append(ex)
        self.assertIn(ex, exception_group.exceptions)

    def test_str_representation_aggregates_all_exceptions(self):
        ex1 = Exception('ex 1')
        ex2 = Exception('ex 2')
        exception_group = scheduler.ExceptionGroup([ex1, ex2])
        self.assertEqual("['ex 1', 'ex 2']", str(exception_group))