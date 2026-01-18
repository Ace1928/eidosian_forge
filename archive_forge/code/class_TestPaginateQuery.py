from unittest import mock
from urllib import parse
import fixtures
import sqlalchemy
from sqlalchemy import Boolean, Index, Integer, DateTime, String
from sqlalchemy import MetaData, Table, Column
from sqlalchemy import ForeignKey, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import psycopg2
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import column_property
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import sql
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql import select
from sqlalchemy.types import UserDefinedType
from oslo_db import exception
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class TestPaginateQuery(test_base.BaseTestCase):

    def setUp(self):
        super(TestPaginateQuery, self).setUp()
        self.query = mock.Mock()
        self.mock_asc = self.useFixture(fixtures.MockPatchObject(sqlalchemy, 'asc')).mock
        self.mock_desc = self.useFixture(fixtures.MockPatchObject(sqlalchemy, 'desc')).mock
        self.marker = FakeTable(user_id='user', project_id='p', snapshot_id='s', updated_at=None)
        self.model = FakeTable

    def test_paginate_query_no_pagination_no_sort_dirs(self):
        self.query.order_by.return_value = self.query
        self.mock_asc.side_effect = ['asc_3', 'asc_2', 'asc_1']
        utils.paginate_query(self.query, self.model, 5, ['user_id', 'project_id', 'snapshot_id'])
        self.mock_asc.assert_has_calls([mock.call(self.model.user_id), mock.call(self.model.project_id), mock.call(self.model.snapshot_id)])
        self.query.order_by.assert_has_calls([mock.call('asc_3'), mock.call('asc_2'), mock.call('asc_1')])
        self.query.limit.assert_called_once_with(5)

    def test_paginate_query_no_pagination(self):
        self.query.order_by.return_value = self.query
        self.mock_asc.side_effect = ['asc']
        self.mock_desc.side_effect = ['desc']
        utils.paginate_query(self.query, self.model, 5, ['user_id', 'project_id'], sort_dirs=['asc', 'desc'])
        self.mock_asc.assert_called_once_with(self.model.user_id)
        self.mock_desc.assert_called_once_with(self.model.project_id)
        self.query.order_by.assert_has_calls([mock.call('asc'), mock.call('desc')])
        self.query.limit.assert_called_once_with(5)

    def test_invalid_sort_key_str(self):
        self.assertEqual('Sort key supplied is invalid: None', str(exception.InvalidSortKey()))
        self.assertEqual('Sort key supplied is invalid: lol', str(exception.InvalidSortKey('lol')))

    def test_invalid_unicode_paramater_str(self):
        self.assertEqual("Invalid Parameter: Encoding directive wasn't provided.", str(exception.DBInvalidUnicodeParameter()))

    def test_paginate_query_attribute_error(self):
        self.mock_asc.return_value = 'asc'
        self.assertRaises(exception.InvalidSortKey, utils.paginate_query, self.query, self.model, 5, ['user_id', 'non-existent key'])
        self.mock_asc.assert_called_once_with(self.model.user_id)
        self.query.order_by.assert_called_once_with('asc')

    def test_paginate_query_attribute_error_invalid_sortkey(self):
        self.assertRaises(exception.InvalidSortKey, utils.paginate_query, self.query, self.model, 5, ['bad_user_id'])

    def test_paginate_query_attribute_error_invalid_sortkey_2(self):
        self.assertRaises(exception.InvalidSortKey, utils.paginate_query, self.query, self.model, 5, ['foo'])

    def test_paginate_query_attribute_error_invalid_sortkey_3(self):
        self.assertRaises(exception.InvalidSortKey, utils.paginate_query, self.query, self.model, 5, ['asc-nullinvalid'])

    def test_paginate_query_assertion_error(self):
        self.assertRaises(AssertionError, utils.paginate_query, self.query, self.model, 5, ['user_id'], marker=self.marker, sort_dir='asc', sort_dirs=['asc'])

    def test_paginate_query_assertion_error_2(self):
        self.assertRaises(AssertionError, utils.paginate_query, self.query, self.model, 5, ['user_id'], marker=self.marker, sort_dir=None, sort_dirs=['asc', 'desk'])

    @mock.patch.object(sqlalchemy.sql, 'and_')
    @mock.patch.object(sqlalchemy.sql, 'or_')
    def test_paginate_query(self, mock_or, mock_and):
        self.query.order_by.return_value = self.query
        self.query.filter.return_value = self.query
        self.mock_asc.return_value = 'asc_1'
        self.mock_desc.return_value = 'desc_1'
        mock_and.side_effect = ['some_crit', 'another_crit']
        mock_or.return_value = 'some_f'
        utils.paginate_query(self.query, self.model, 5, ['user_id', 'project_id'], marker=self.marker, sort_dirs=['asc', 'desc'])
        self.mock_asc.assert_called_once_with(self.model.user_id)
        self.mock_desc.assert_called_once_with(self.model.project_id)
        self.query.order_by.assert_has_calls([mock.call('asc_1'), mock.call('desc_1')])
        mock_and.assert_has_calls([mock.call(mock.ANY), mock.call(mock.ANY, mock.ANY)])
        mock_or.assert_called_once_with('some_crit', 'another_crit')
        self.query.filter.assert_called_once_with('some_f')
        self.query.limit.assert_called_once_with(5)

    @mock.patch.object(sqlalchemy.sql, 'and_')
    @mock.patch.object(sqlalchemy.sql, 'or_')
    def test_paginate_query_null(self, mock_or, mock_and):
        self.query.order_by.return_value = self.query
        self.query.filter.return_value = self.query
        self.mock_desc.side_effect = ['asc_null_2', 'desc_null_2', 'desc_1']
        self.mock_asc.side_effect = ['asc_1']
        mock_or.side_effect = ['or_1', 'or_2', 'some_f']
        mock_and.side_effect = ['some_crit', 'another_crit']
        with mock.patch.object(self.model.user_id.comparator.expression, 'is_not') as mock_is_not, mock.patch.object(self.model.user_id.comparator.expression, 'is_') as mock_is_a, mock.patch.object(self.model.project_id.comparator.expression, 'is_') as mock_is_b:
            mock_is_not.return_value = 'asc_null_1'
            mock_is_a.side_effect = ['desc_null_filter_1', 'desc_null_filter_2']
            mock_is_b.side_effect = ['desc_null_1', 'asc_null_filter']
            utils.paginate_query(self.query, self.model, 5, ['user_id', 'project_id'], marker=self.marker, sort_dirs=['asc-nullslast', 'desc-nullsfirst'])
            mock_is_not.assert_called_once_with(None)
            mock_is_a.assert_has_calls([mock.call(None), mock.call(None)])
            mock_is_b.assert_has_calls([mock.call(None), mock.call(None)])
        self.mock_desc.assert_has_calls([mock.call('asc_null_1'), mock.call('desc_null_1'), mock.call(self.model.project_id)])
        self.mock_asc.assert_has_calls([mock.call(self.model.user_id)])
        mock_or.assert_has_calls([mock.call(mock.ANY, 'desc_null_filter_2'), mock.call(mock.ANY, 'asc_null_filter'), mock.call('some_crit', 'another_crit')])
        mock_and.assert_has_calls([mock.call('or_1'), mock.call(mock.ANY, 'or_2')])
        self.query.order_by.assert_has_calls([mock.call('asc_null_2'), mock.call('asc_1'), mock.call('desc_null_2'), mock.call('desc_1')])
        self.query.filter.assert_called_once_with('some_f')
        self.query.limit.assert_called_once_with(5)

    @mock.patch.object(sqlalchemy.sql, 'and_')
    @mock.patch.object(sqlalchemy.sql, 'or_')
    def test_paginate_query_marker_null(self, mock_or, mock_and):
        self.mock_asc.side_effect = ['asc_1']
        self.mock_desc.side_effect = ['asc_null_2', 'desc_null_2', 'desc_1']
        self.query.order_by.return_value = self.query
        self.query.filter.return_value = self.query
        mock_and.return_value = 'some_crit'
        mock_or.side_effect = ['or_1', 'some_f']
        with mock.patch.object(self.model.user_id.comparator.expression, 'is_not') as mock_is_not, mock.patch.object(self.model.updated_at.comparator.expression, 'is_') as mock_is_a, mock.patch.object(self.model.user_id.comparator.expression, 'is_') as mock_is_b:
            mock_is_not.return_value = 'asc_null_1'
            mock_is_a.return_value = 'desc_null_1'
            mock_is_b.side_effect = ['asc_null_filter_1', 'asc_null_filter_2']
            utils.paginate_query(self.query, self.model, 5, ['user_id', 'updated_at'], marker=self.marker, sort_dirs=['asc-nullslast', 'desc-nullsfirst'])
            mock_is_not.assert_called_once_with(None)
            mock_is_a.assert_called_once_with(None)
            mock_is_b.assert_has_calls([mock.call(None), mock.call(None)])
        self.mock_asc.assert_called_once_with(self.model.user_id)
        self.mock_desc.assert_has_calls([mock.call('asc_null_1'), mock.call('desc_null_1'), mock.call(self.model.updated_at)])
        mock_and.assert_called_once_with('or_1')
        mock_or.assert_has_calls([mock.call(mock.ANY, 'asc_null_filter_2'), mock.call('some_crit')])
        self.query.order_by.assert_has_calls([mock.call('asc_null_2'), mock.call('asc_1'), mock.call('desc_null_2'), mock.call('desc_1')])
        self.query.filter.assert_called_once_with('some_f')
        self.query.limit.assert_called_once_with(5)

    @mock.patch.object(sqlalchemy.sql, 'and_')
    @mock.patch.object(sqlalchemy.sql, 'or_')
    def test_paginate_query_marker_null_with_two_primary_keys(self, mock_or, mock_and):
        self.mock_asc.return_value = 'asc_1'
        self.mock_desc.side_effect = ['asc_null_2', 'desc_null_2', 'desc_1', 'desc_null_4', 'desc_4']
        self.query.order_by.return_value = self.query
        mock_or.side_effect = ['or_1', 'or_2', 'some_f']
        mock_and.side_effect = ['some_crit', 'other_crit']
        self.query.filter.return_value = self.query
        with mock.patch.object(self.model.user_id.comparator.expression, 'is_not') as mock_is_not, mock.patch.object(self.model.updated_at.comparator.expression, 'is_') as mock_is_a, mock.patch.object(self.model.user_id.comparator.expression, 'is_') as mock_is_b, mock.patch.object(self.model.project_id.comparator.expression, 'is_') as mock_is_c:
            mock_is_not.return_value = 'asc_null_1'
            mock_is_a.return_value = 'desc_null_1'
            mock_is_b.side_effect = ['asc_null_filter_1', 'asc_null_filter_2']
            mock_is_c.side_effect = ['desc_null_3', 'desc_null_filter_3']
            utils.paginate_query(self.query, self.model, 5, ['user_id', 'updated_at', 'project_id'], marker=self.marker, sort_dirs=['asc-nullslast', 'desc-nullsfirst', 'desc-nullsfirst'])
            mock_is_not.assert_called_once_with(None)
            mock_is_a.assert_called_once_with(None)
            mock_is_b.assert_has_calls([mock.call(None), mock.call(None)])
            mock_is_c.assert_has_calls([mock.call(None), mock.call(None)])
        self.mock_asc.assert_called_once_with(self.model.user_id)
        self.mock_desc.assert_has_calls([mock.call('asc_null_1'), mock.call('desc_null_1'), mock.call(self.model.updated_at), mock.call('desc_null_3'), mock.call(self.model.project_id)])
        self.query.order_by.assert_has_calls([mock.call('asc_null_2'), mock.call('asc_1'), mock.call('desc_null_2'), mock.call('desc_1'), mock.call('desc_null_4'), mock.call('desc_4')])
        mock_or.assert_has_calls([mock.call(mock.ANY, 'asc_null_filter_2'), mock.call(mock.ANY, 'desc_null_filter_3'), mock.call('some_crit', 'other_crit')])
        mock_and.assert_has_calls([mock.call('or_1'), mock.call(mock.ANY, 'or_2')])
        self.query.filter.assert_called_once_with('some_f')
        self.query.limit.assert_called_once_with(5)

    def test_paginate_query_value_error(self):
        self.mock_asc.return_value = 'asc_1'
        self.query.order_by.return_value = self.query
        self.assertRaises(ValueError, utils.paginate_query, self.query, self.model, 5, ['user_id', 'project_id'], marker=self.marker, sort_dirs=['asc', 'mixed'])
        self.mock_asc.assert_called_once_with(self.model.user_id)
        self.query.order_by.assert_called_once_with('asc_1')

    def test_paginate_on_hybrid(self):
        self.mock_asc.return_value = 'asc_1'
        self.mock_desc.return_value = 'desc_1'
        self.query.order_by.return_value = self.query
        utils.paginate_query(self.query, self.model, 5, ['user_id', 'some_hybrid'], sort_dirs=['asc', 'desc'])
        self.mock_asc.assert_called_once_with(self.model.user_id)
        self.mock_desc.assert_called_once_with(self.model.some_hybrid)
        self.query.order_by.assert_has_calls([mock.call('asc_1'), mock.call('desc_1')])
        self.query.limit.assert_called_once_with(5)