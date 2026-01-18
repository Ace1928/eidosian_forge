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
class JSONTest(_LiteralRoundTripFixture, fixtures.TablesTest):
    __requires__ = ('json_type',)
    __backend__ = True
    datatype = JSON

    @classmethod
    def define_tables(cls, metadata):
        Table('data_table', metadata, Column('id', Integer, primary_key=True), Column('name', String(30), nullable=False), Column('data', cls.datatype, nullable=False), Column('nulldata', cls.datatype(none_as_null=True)))

    def test_round_trip_data1(self, connection):
        self._test_round_trip({'key1': 'value1', 'key2': 'value2'}, connection)

    @testing.combinations(('unicode', True), ('ascii', False), argnames='unicode_', id_='ia')
    @testing.combinations(100, 1999, 3000, 4000, 5000, 9000, argnames='length')
    def test_round_trip_pretty_large_data(self, connection, unicode_, length):
        if unicode_:
            data = 'réve🐍illé' * (length // 9 + 1)
            data = data[0:length // 2]
        else:
            data = 'abcdefg' * (length // 7 + 1)
            data = data[0:length]
        self._test_round_trip({'key1': data, 'key2': data}, connection)

    def _test_round_trip(self, data_element, connection):
        data_table = self.tables.data_table
        connection.execute(data_table.insert(), {'id': 1, 'name': 'row1', 'data': data_element})
        row = connection.execute(select(data_table.c.data)).first()
        eq_(row, (data_element,))

    def _index_fixtures(include_comparison):
        if include_comparison:
            json_elements = []
        else:
            json_elements = [('json', {'foo': 'bar'}), ('json', ['one', 'two', 'three']), (None, {'foo': 'bar'}), (None, ['one', 'two', 'three'])]
        elements = [('boolean', True), ('boolean', False), ('boolean', None), ('string', 'some string'), ('string', None), ('string', 'réve illé'), ('string', 'réve🐍 illé', testing.requires.json_index_supplementary_unicode_element), ('integer', 15), ('integer', 1), ('integer', 0), ('integer', None), ('float', 28.5), ('float', None), ('float', 1234567.89, testing.requires.literal_float_coercion), ('numeric', 1234567.89), ('numeric', 99998969694839.98), ('numeric', 99939.983485848), ('_decimal', decimal.Decimal('1234567.89')), ('_decimal', decimal.Decimal('99998969694839.983485848'), requirements.cast_precision_numerics_many_significant_digits), ('_decimal', decimal.Decimal('99939.983485848'))] + json_elements

        def decorate(fn):
            fn = testing.combinations(*elements, id_='sa')(fn)
            return fn
        return decorate

    def _json_value_insert(self, connection, datatype, value, data_element):
        data_table = self.tables.data_table
        if datatype == '_decimal':

            class DecimalEncoder(json.JSONEncoder):

                def default(self, o):
                    if isinstance(o, decimal.Decimal):
                        return str(o)
                    return super().default(o)
            json_data = json.dumps(data_element, cls=DecimalEncoder)
            json_data = re.sub('"(%s)"' % str(value), str(value), json_data)
            datatype = 'numeric'
            connection.execute(data_table.insert().values(name='row1', data=bindparam(None, json_data, literal_execute=True), nulldata=bindparam(None, json_data, literal_execute=True)))
        else:
            connection.execute(data_table.insert(), {'name': 'row1', 'data': data_element, 'nulldata': data_element})
        p_s = None
        if datatype:
            if datatype == 'numeric':
                a, b = str(value).split('.')
                s = len(b)
                p = len(a) + s
                if isinstance(value, decimal.Decimal):
                    compare_value = value
                else:
                    compare_value = decimal.Decimal(str(value))
                p_s = (p, s)
            else:
                compare_value = value
        else:
            compare_value = value
        return (datatype, compare_value, p_s)

    @_index_fixtures(False)
    def test_index_typed_access(self, datatype, value):
        data_table = self.tables.data_table
        data_element = {'key1': value}
        with config.db.begin() as conn:
            datatype, compare_value, p_s = self._json_value_insert(conn, datatype, value, data_element)
            expr = data_table.c.data['key1']
            if datatype:
                if datatype == 'numeric' and p_s:
                    expr = expr.as_numeric(*p_s)
                else:
                    expr = getattr(expr, 'as_%s' % datatype)()
            roundtrip = conn.scalar(select(expr))
            eq_(roundtrip, compare_value)
            is_(type(roundtrip), type(compare_value))

    @_index_fixtures(True)
    def test_index_typed_comparison(self, datatype, value):
        data_table = self.tables.data_table
        data_element = {'key1': value}
        with config.db.begin() as conn:
            datatype, compare_value, p_s = self._json_value_insert(conn, datatype, value, data_element)
            expr = data_table.c.data['key1']
            if datatype:
                if datatype == 'numeric' and p_s:
                    expr = expr.as_numeric(*p_s)
                else:
                    expr = getattr(expr, 'as_%s' % datatype)()
            row = conn.execute(select(expr).where(expr == compare_value)).first()
            eq_(row, (compare_value,))

    @_index_fixtures(True)
    def test_path_typed_comparison(self, datatype, value):
        data_table = self.tables.data_table
        data_element = {'key1': {'subkey1': value}}
        with config.db.begin() as conn:
            datatype, compare_value, p_s = self._json_value_insert(conn, datatype, value, data_element)
            expr = data_table.c.data['key1', 'subkey1']
            if datatype:
                if datatype == 'numeric' and p_s:
                    expr = expr.as_numeric(*p_s)
                else:
                    expr = getattr(expr, 'as_%s' % datatype)()
            row = conn.execute(select(expr).where(expr == compare_value)).first()
            eq_(row, (compare_value,))

    @testing.combinations((True,), (False,), (None,), (15,), (0,), (-1,), (-1.0,), (15.052,), ('a string',), ('réve illé',), ('réve🐍 illé',))
    def test_single_element_round_trip(self, element):
        data_table = self.tables.data_table
        data_element = element
        with config.db.begin() as conn:
            conn.execute(data_table.insert(), {'name': 'row1', 'data': data_element, 'nulldata': data_element})
            row = conn.execute(select(data_table.c.data, data_table.c.nulldata)).first()
            eq_(row, (data_element, data_element))

    def test_round_trip_custom_json(self):
        data_table = self.tables.data_table
        data_element = {'key1': 'data1'}
        js = mock.Mock(side_effect=json.dumps)
        jd = mock.Mock(side_effect=json.loads)
        engine = engines.testing_engine(options=dict(json_serializer=js, json_deserializer=jd))
        data_table.create(engine, checkfirst=True)
        with engine.begin() as conn:
            conn.execute(data_table.insert(), {'name': 'row1', 'data': data_element})
            row = conn.execute(select(data_table.c.data)).first()
            eq_(row, (data_element,))
            eq_(js.mock_calls, [mock.call(data_element)])
            if testing.requires.json_deserializer_binary.enabled:
                eq_(jd.mock_calls, [mock.call(json.dumps(data_element).encode())])
            else:
                eq_(jd.mock_calls, [mock.call(json.dumps(data_element))])

    @testing.combinations(('parameters',), ('multiparameters',), ('values',), ('omit',), argnames='insert_type')
    def test_round_trip_none_as_sql_null(self, connection, insert_type):
        col = self.tables.data_table.c['nulldata']
        conn = connection
        if insert_type == 'parameters':
            stmt, params = (self.tables.data_table.insert(), {'name': 'r1', 'nulldata': None, 'data': None})
        elif insert_type == 'multiparameters':
            stmt, params = (self.tables.data_table.insert(), [{'name': 'r1', 'nulldata': None, 'data': None}])
        elif insert_type == 'values':
            stmt, params = (self.tables.data_table.insert().values(name='r1', nulldata=None, data=None), {})
        elif insert_type == 'omit':
            stmt, params = (self.tables.data_table.insert(), {'name': 'r1', 'data': None})
        else:
            assert False
        conn.execute(stmt, params)
        eq_(conn.scalar(select(self.tables.data_table.c.name).where(col.is_(null()))), 'r1')
        eq_(conn.scalar(select(col)), None)

    def test_round_trip_json_null_as_json_null(self, connection):
        col = self.tables.data_table.c['data']
        conn = connection
        conn.execute(self.tables.data_table.insert(), {'name': 'r1', 'data': JSON.NULL})
        eq_(conn.scalar(select(self.tables.data_table.c.name).where(cast(col, String) == 'null')), 'r1')
        eq_(conn.scalar(select(col)), None)

    @testing.combinations(('parameters',), ('multiparameters',), ('values',), argnames='insert_type')
    def test_round_trip_none_as_json_null(self, connection, insert_type):
        col = self.tables.data_table.c['data']
        if insert_type == 'parameters':
            stmt, params = (self.tables.data_table.insert(), {'name': 'r1', 'data': None})
        elif insert_type == 'multiparameters':
            stmt, params = (self.tables.data_table.insert(), [{'name': 'r1', 'data': None}])
        elif insert_type == 'values':
            stmt, params = (self.tables.data_table.insert().values(name='r1', data=None), {})
        else:
            assert False
        conn = connection
        conn.execute(stmt, params)
        eq_(conn.scalar(select(self.tables.data_table.c.name).where(cast(col, String) == 'null')), 'r1')
        eq_(conn.scalar(select(col)), None)

    def test_unicode_round_trip(self):
        with config.db.begin() as conn:
            conn.execute(self.tables.data_table.insert(), {'name': 'r1', 'data': {'réve🐍 illé': 'réve🐍 illé', 'data': {'k1': 'drôl🐍e'}}})
            eq_(conn.scalar(select(self.tables.data_table.c.data)), {'réve🐍 illé': 'réve🐍 illé', 'data': {'k1': 'drôl🐍e'}})

    def test_eval_none_flag_orm(self, connection):
        Base = declarative_base()

        class Data(Base):
            __table__ = self.tables.data_table
        with Session(connection) as s:
            d1 = Data(name='d1', data=None, nulldata=None)
            s.add(d1)
            s.commit()
            s.bulk_insert_mappings(Data, [{'name': 'd2', 'data': None, 'nulldata': None}])
            eq_(s.query(cast(self.tables.data_table.c.data, String()), cast(self.tables.data_table.c.nulldata, String)).filter(self.tables.data_table.c.name == 'd1').first(), ('null', None))
            eq_(s.query(cast(self.tables.data_table.c.data, String()), cast(self.tables.data_table.c.nulldata, String)).filter(self.tables.data_table.c.name == 'd2').first(), ('null', None))