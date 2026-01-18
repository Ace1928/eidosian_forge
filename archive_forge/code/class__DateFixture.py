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
class _DateFixture(_LiteralRoundTripFixture, fixtures.TestBase):
    compare = None

    @classmethod
    def define_tables(cls, metadata):

        class Decorated(TypeDecorator):
            impl = cls.datatype
            cache_ok = True
        Table('date_table', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('date_data', cls.datatype), Column('decorated_date_data', Decorated))

    def test_round_trip(self, connection):
        date_table = self.tables.date_table
        connection.execute(date_table.insert(), {'id': 1, 'date_data': self.data})
        row = connection.execute(select(date_table.c.date_data)).first()
        compare = self.compare or self.data
        eq_(row, (compare,))
        assert isinstance(row[0], type(compare))

    def test_round_trip_decorated(self, connection):
        date_table = self.tables.date_table
        connection.execute(date_table.insert(), {'id': 1, 'decorated_date_data': self.data})
        row = connection.execute(select(date_table.c.decorated_date_data)).first()
        compare = self.compare or self.data
        eq_(row, (compare,))
        assert isinstance(row[0], type(compare))

    def test_null(self, connection):
        date_table = self.tables.date_table
        connection.execute(date_table.insert(), {'id': 1, 'date_data': None})
        row = connection.execute(select(date_table.c.date_data)).first()
        eq_(row, (None,))

    @testing.requires.datetime_literals
    def test_literal(self, literal_round_trip):
        compare = self.compare or self.data
        literal_round_trip(self.datatype, [self.data], [compare], compare=compare)

    @testing.requires.standalone_null_binds_whereclause
    def test_null_bound_comparison(self):
        date_table = self.tables.date_table
        with config.db.begin() as conn:
            result = conn.execute(date_table.insert(), {'id': 1, 'date_data': self.data})
            id_ = result.inserted_primary_key[0]
            stmt = select(date_table.c.id).where(case((bindparam('foo', type_=self.datatype) != None, bindparam('foo', type_=self.datatype)), else_=date_table.c.date_data) == date_table.c.date_data)
            row = conn.execute(stmt, {'foo': None}).first()
            eq_(row[0], id_)