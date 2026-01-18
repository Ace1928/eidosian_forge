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
class _LiteralRoundTripFixture:
    supports_whereclause = True

    @testing.fixture
    def literal_round_trip(self, metadata, connection):
        """test literal rendering"""

        def run(type_, input_, output, filter_=None, compare=None, support_whereclause=True):
            t = Table('t', metadata, Column('x', type_))
            t.create(connection)
            for value in input_:
                ins = t.insert().values(x=literal(value, type_, literal_execute=True))
                connection.execute(ins)
            ins = t.insert().values(x=literal(None, type_, literal_execute=True))
            connection.execute(ins)
            if support_whereclause and self.supports_whereclause:
                if compare:
                    stmt = t.select().where(t.c.x == literal(compare, type_, literal_execute=True), t.c.x == literal(input_[0], type_, literal_execute=True))
                else:
                    stmt = t.select().where(t.c.x == literal(compare if compare is not None else input_[0], type_, literal_execute=True))
            else:
                stmt = t.select().where(t.c.x.is_not(None))
            rows = connection.execute(stmt).all()
            assert rows, 'No rows returned'
            for row in rows:
                value = row[0]
                if filter_ is not None:
                    value = filter_(value)
                assert value in output
            stmt = t.select().where(t.c.x.is_(None))
            rows = connection.execute(stmt).all()
            eq_(rows, [(None,)])
        return run