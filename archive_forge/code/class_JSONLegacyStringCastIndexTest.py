import datetime
import decimal
import json
import re
import uuid
from .. import config
from .. import engines
from .. import fixtures
from .. import mock
from ..assertions import eq_
from ..assertions import is_
from ..assertions import ne_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import and_
from ... import ARRAY
from ... import BigInteger
from ... import bindparam
from ... import Boolean
from ... import case
from ... import cast
from ... import Date
from ... import DateTime
from ... import Float
from ... import Integer
from ... import Interval
from ... import JSON
from ... import literal
from ... import literal_column
from ... import MetaData
from ... import null
from ... import Numeric
from ... import select
from ... import String
from ... import testing
from ... import Text
from ... import Time
from ... import TIMESTAMP
from ... import type_coerce
from ... import TypeDecorator
from ... import Unicode
from ... import UnicodeText
from ... import UUID
from ... import Uuid
from ...orm import declarative_base
from ...orm import Session
from ...sql import sqltypes
from ...sql.sqltypes import LargeBinary
from ...sql.sqltypes import PickleType
class JSONLegacyStringCastIndexTest(_LiteralRoundTripFixture, fixtures.TablesTest):
    """test JSON index access with "cast to string", which we have documented
    for a long time as how to compare JSON values, but is ultimately not
    reliable in all cases.   The "as_XYZ()" comparators should be used
    instead.

    """
    __requires__ = ('json_type', 'legacy_unconditional_json_extract')
    __backend__ = True
    datatype = JSON
    data1 = {'key1': 'value1', 'key2': 'value2'}
    data2 = {"Key 'One'": 'value1', 'key two': 'value2', 'key three': "value ' three '"}
    data3 = {'key1': [1, 2, 3], 'key2': ['one', 'two', 'three'], 'key3': [{'four': 'five'}, {'six': 'seven'}]}
    data4 = ['one', 'two', 'three']
    data5 = {'nested': {'elem1': [{'a': 'b', 'c': 'd'}, {'e': 'f', 'g': 'h'}], 'elem2': {'elem3': {'elem4': 'elem5'}}}}
    data6 = {'a': 5, 'b': 'some value', 'c': {'foo': 'bar'}}

    @classmethod
    def define_tables(cls, metadata):
        Table('data_table', metadata, Column('id', Integer, primary_key=True), Column('name', String(30), nullable=False), Column('data', cls.datatype), Column('nulldata', cls.datatype(none_as_null=True)))

    def _criteria_fixture(self):
        with config.db.begin() as conn:
            conn.execute(self.tables.data_table.insert(), [{'name': 'r1', 'data': self.data1}, {'name': 'r2', 'data': self.data2}, {'name': 'r3', 'data': self.data3}, {'name': 'r4', 'data': self.data4}, {'name': 'r5', 'data': self.data5}, {'name': 'r6', 'data': self.data6}])

    def _test_index_criteria(self, crit, expected, test_literal=True):
        self._criteria_fixture()
        with config.db.connect() as conn:
            stmt = select(self.tables.data_table.c.name).where(crit)
            eq_(conn.scalar(stmt), expected)
            if test_literal:
                literal_sql = str(stmt.compile(config.db, compile_kwargs={'literal_binds': True}))
                eq_(conn.exec_driver_sql(literal_sql).scalar(), expected)

    def test_string_cast_crit_spaces_in_key(self):
        name = self.tables.data_table.c.name
        col = self.tables.data_table.c['data']
        self._test_index_criteria(and_(name.in_(['r1', 'r2', 'r3']), cast(col['key two'], String) == '"value2"'), 'r2')

    @config.requirements.json_array_indexes
    def test_string_cast_crit_simple_int(self):
        name = self.tables.data_table.c.name
        col = self.tables.data_table.c['data']
        self._test_index_criteria(and_(name == 'r4', cast(col[1], String) == '"two"'), 'r4')

    def test_string_cast_crit_mixed_path(self):
        col = self.tables.data_table.c['data']
        self._test_index_criteria(cast(col['key3', 1, 'six'], String) == '"seven"', 'r3')

    def test_string_cast_crit_string_path(self):
        col = self.tables.data_table.c['data']
        self._test_index_criteria(cast(col['nested', 'elem2', 'elem3', 'elem4'], String) == '"elem5"', 'r5')

    def test_string_cast_crit_against_string_basic(self):
        name = self.tables.data_table.c.name
        col = self.tables.data_table.c['data']
        self._test_index_criteria(and_(name == 'r6', cast(col['b'], String) == '"some value"'), 'r6')