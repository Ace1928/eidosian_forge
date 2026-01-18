from sqlalchemy.dialects.mysql import base as mysql_base
from sqlalchemy.dialects.sqlite import base as sqlite_base
from sqlalchemy import types
from heat.db import types as db_types
from heat.tests import common
class ListTest(common.HeatTestCase):

    def setUp(self):
        super(ListTest, self).setUp()
        self.sqltype = db_types.List()

    def test_load_dialect_impl(self):
        dialect = mysql_base.MySQLDialect()
        impl = self.sqltype.load_dialect_impl(dialect)
        self.assertNotEqual(types.Text, type(impl))
        dialect = sqlite_base.SQLiteDialect()
        impl = self.sqltype.load_dialect_impl(dialect)
        self.assertEqual(types.Text, type(impl))

    def test_process_bind_param(self):
        dialect = None
        value = ['foo', 'bar']
        result = self.sqltype.process_bind_param(value, dialect)
        self.assertEqual('["foo", "bar"]', result)

    def test_process_bind_param_null(self):
        dialect = None
        value = None
        result = self.sqltype.process_bind_param(value, dialect)
        self.assertEqual('null', result)

    def test_process_result_value(self):
        dialect = None
        value = '["foo", "bar"]'
        result = self.sqltype.process_result_value(value, dialect)
        self.assertEqual(['foo', 'bar'], result)

    def test_process_result_value_null(self):
        dialect = None
        value = None
        result = self.sqltype.process_result_value(value, dialect)
        self.assertIsNone(result)