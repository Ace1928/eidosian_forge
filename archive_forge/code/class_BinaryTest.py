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
class BinaryTest(_LiteralRoundTripFixture, fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('binary_table', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('binary_data', LargeBinary), Column('pickle_data', PickleType))

    @testing.combinations(b'this is binary', b'7\xe7\x9f', argnames='data')
    def test_binary_roundtrip(self, connection, data):
        binary_table = self.tables.binary_table
        connection.execute(binary_table.insert(), {'id': 1, 'binary_data': data})
        row = connection.execute(select(binary_table.c.binary_data)).first()
        eq_(row, (data,))

    def test_pickle_roundtrip(self, connection):
        binary_table = self.tables.binary_table
        connection.execute(binary_table.insert(), {'id': 1, 'pickle_data': {'foo': [1, 2, 3], 'bar': 'bat'}})
        row = connection.execute(select(binary_table.c.pickle_data)).first()
        eq_(row, ({'foo': [1, 2, 3], 'bar': 'bat'},))