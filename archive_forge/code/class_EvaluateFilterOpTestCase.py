import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
class EvaluateFilterOpTestCase(test_utils.BaseTestCase):

    def test_less_than_operator(self):
        self.assertTrue(utils.evaluate_filter_op(9, 'lt', 10))
        self.assertFalse(utils.evaluate_filter_op(10, 'lt', 10))
        self.assertFalse(utils.evaluate_filter_op(11, 'lt', 10))

    def test_less_than_equal_operator(self):
        self.assertTrue(utils.evaluate_filter_op(9, 'lte', 10))
        self.assertTrue(utils.evaluate_filter_op(10, 'lte', 10))
        self.assertFalse(utils.evaluate_filter_op(11, 'lte', 10))

    def test_greater_than_operator(self):
        self.assertFalse(utils.evaluate_filter_op(9, 'gt', 10))
        self.assertFalse(utils.evaluate_filter_op(10, 'gt', 10))
        self.assertTrue(utils.evaluate_filter_op(11, 'gt', 10))

    def test_greater_than_equal_operator(self):
        self.assertFalse(utils.evaluate_filter_op(9, 'gte', 10))
        self.assertTrue(utils.evaluate_filter_op(10, 'gte', 10))
        self.assertTrue(utils.evaluate_filter_op(11, 'gte', 10))

    def test_not_equal_operator(self):
        self.assertTrue(utils.evaluate_filter_op(9, 'neq', 10))
        self.assertFalse(utils.evaluate_filter_op(10, 'neq', 10))
        self.assertTrue(utils.evaluate_filter_op(11, 'neq', 10))

    def test_equal_operator(self):
        self.assertFalse(utils.evaluate_filter_op(9, 'eq', 10))
        self.assertTrue(utils.evaluate_filter_op(10, 'eq', 10))
        self.assertFalse(utils.evaluate_filter_op(11, 'eq', 10))

    def test_invalid_operator(self):
        self.assertRaises(exception.InvalidFilterOperatorValue, utils.evaluate_filter_op, '10', 'bar', '8')