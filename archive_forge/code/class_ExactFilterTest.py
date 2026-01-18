from unittest import mock
from heat.db import filters as db_filters
from heat.tests import common
class ExactFilterTest(common.HeatTestCase):

    def setUp(self):
        super(ExactFilterTest, self).setUp()
        self.query = mock.Mock()
        self.model = mock.Mock()

    def test_returns_same_query_for_empty_filters(self):
        filters = {}
        db_filters.exact_filter(self.query, self.model, filters)
        self.assertEqual(0, self.query.call_count)

    def test_add_exact_match_clause_for_single_values(self):
        filters = {'cat': 'foo'}
        db_filters.exact_filter(self.query, self.model, filters)
        self.query.filter_by.assert_called_once_with(cat='foo')

    def test_adds_an_in_clause_for_multiple_values(self):
        self.model.cat.in_.return_value = 'fake in clause'
        filters = {'cat': ['foo', 'quux']}
        db_filters.exact_filter(self.query, self.model, filters)
        self.query.filter.assert_called_once_with('fake in clause')
        self.model.cat.in_.assert_called_once_with(['foo', 'quux'])