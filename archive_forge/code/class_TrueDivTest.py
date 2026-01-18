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
class TrueDivTest(fixtures.TestBase):
    __backend__ = True

    @testing.combinations(('15', '10', 1.5), ('-15', '10', -1.5), argnames='left, right, expected')
    def test_truediv_integer(self, connection, left, right, expected):
        """test #4926"""
        eq_(connection.scalar(select(literal_column(left, type_=Integer()) / literal_column(right, type_=Integer()))), expected)

    @testing.combinations(('15', '10', 1), ('-15', '5', -3), argnames='left, right, expected')
    def test_floordiv_integer(self, connection, left, right, expected):
        """test #4926"""
        eq_(connection.scalar(select(literal_column(left, type_=Integer()) // literal_column(right, type_=Integer()))), expected)

    @testing.combinations(('5.52', '2.4', '2.3'), argnames='left, right, expected')
    def test_truediv_numeric(self, connection, left, right, expected):
        """test #4926"""
        eq_(connection.scalar(select(literal_column(left, type_=Numeric(10, 2)) / literal_column(right, type_=Numeric(10, 2)))), decimal.Decimal(expected))

    @testing.combinations(('5.52', '2.4', 2.3), argnames='left, right, expected')
    def test_truediv_float(self, connection, left, right, expected):
        """test #4926"""
        eq_(connection.scalar(select(literal_column(left, type_=Float()) / literal_column(right, type_=Float()))), expected)

    @testing.combinations(('5.52', '2.4', '2.0'), argnames='left, right, expected')
    def test_floordiv_numeric(self, connection, left, right, expected):
        """test #4926"""
        eq_(connection.scalar(select(literal_column(left, type_=Numeric()) // literal_column(right, type_=Numeric()))), decimal.Decimal(expected))

    def test_truediv_integer_bound(self, connection):
        """test #4926"""
        eq_(connection.scalar(select(literal(15) / literal(10))), 1.5)

    def test_floordiv_integer_bound(self, connection):
        """test #4926"""
        eq_(connection.scalar(select(literal(15) // literal(10))), 1)