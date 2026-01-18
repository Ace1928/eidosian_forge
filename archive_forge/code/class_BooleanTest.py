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
class BooleanTest(_LiteralRoundTripFixture, fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('boolean_table', metadata, Column('id', Integer, primary_key=True, autoincrement=False), Column('value', Boolean), Column('unconstrained_value', Boolean(create_constraint=False)))

    def test_render_literal_bool(self, literal_round_trip):
        literal_round_trip(Boolean(), [True, False], [True, False])

    def test_round_trip(self, connection):
        boolean_table = self.tables.boolean_table
        connection.execute(boolean_table.insert(), {'id': 1, 'value': True, 'unconstrained_value': False})
        row = connection.execute(select(boolean_table.c.value, boolean_table.c.unconstrained_value)).first()
        eq_(row, (True, False))
        assert isinstance(row[0], bool)

    @testing.requires.nullable_booleans
    def test_null(self, connection):
        boolean_table = self.tables.boolean_table
        connection.execute(boolean_table.insert(), {'id': 1, 'value': None, 'unconstrained_value': None})
        row = connection.execute(select(boolean_table.c.value, boolean_table.c.unconstrained_value)).first()
        eq_(row, (None, None))

    def test_whereclause(self):
        boolean_table = self.tables.boolean_table
        with config.db.begin() as conn:
            conn.execute(boolean_table.insert(), [{'id': 1, 'value': True, 'unconstrained_value': True}, {'id': 2, 'value': False, 'unconstrained_value': False}])
            eq_(conn.scalar(select(boolean_table.c.id).where(boolean_table.c.value)), 1)
            eq_(conn.scalar(select(boolean_table.c.id).where(boolean_table.c.unconstrained_value)), 1)
            eq_(conn.scalar(select(boolean_table.c.id).where(~boolean_table.c.value)), 2)
            eq_(conn.scalar(select(boolean_table.c.id).where(~boolean_table.c.unconstrained_value)), 2)