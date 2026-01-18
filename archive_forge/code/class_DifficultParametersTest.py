import importlib
from . import testing
from .. import assert_raises
from .. import config
from .. import engines
from .. import eq_
from .. import fixtures
from .. import is_not_none
from .. import is_true
from .. import ne_
from .. import provide_metadata
from ..assertions import expect_raises
from ..assertions import expect_raises_message
from ..config import requirements
from ..provision import set_default_schema_on_connection
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import dialects
from ... import event
from ... import exc
from ... import Integer
from ... import literal_column
from ... import select
from ... import String
from ...sql.compiler import Compiled
from ...util import inspect_getfullargspec
class DifficultParametersTest(fixtures.TestBase):
    __backend__ = True
    tough_parameters = testing.combinations(('boring',), ('per cent',), ('per % cent',), ('%percent',), ('par(ens)',), ('percent%(ens)yah',), ('col:ons',), ('_starts_with_underscore',), ('dot.s',), ('more :: %colons%',), ('_name',), ('___name',), ('[BracketsAndCase]',), ('42numbers',), ('percent%signs',), ('has spaces',), ('/slashes/',), ('more/slashes',), ('q?marks',), ('1param',), ('1col:on',), argnames='paramname')

    @tough_parameters
    @config.requirements.unusual_column_name_characters
    def test_round_trip_same_named_column(self, paramname, connection, metadata):
        name = paramname
        t = Table('t', metadata, Column('id', Integer, primary_key=True), Column(name, String(50), nullable=False))
        t.create(connection)
        connection.execute(t.insert().values({'id': 1, name: 'some name'}))
        stmt = select(t.c[name]).where(t.c[name] == 'some name')
        eq_(connection.scalar(stmt), 'some name')
        stmt = select(t.c[name]).where(t.c[name] == bindparam(name))
        row = connection.execute(stmt, {name: 'some name'}).first()
        eq_(row._mapping[name], 'some name')
        stmt = select(t.c[name]).where(t.c[name].in_(['some name', 'some other_name']))
        row = connection.execute(stmt).first()

    @testing.fixture
    def multirow_fixture(self, metadata, connection):
        mytable = Table('mytable', metadata, Column('myid', Integer), Column('name', String(50)), Column('desc', String(50)))
        mytable.create(connection)
        connection.execute(mytable.insert(), [{'myid': 1, 'name': 'a', 'desc': 'a_desc'}, {'myid': 2, 'name': 'b', 'desc': 'b_desc'}, {'myid': 3, 'name': 'c', 'desc': 'c_desc'}, {'myid': 4, 'name': 'd', 'desc': 'd_desc'}])
        yield mytable

    @tough_parameters
    def test_standalone_bindparam_escape(self, paramname, connection, multirow_fixture):
        tbl1 = multirow_fixture
        stmt = select(tbl1.c.myid).where(tbl1.c.name == bindparam(paramname, value='x'))
        res = connection.scalar(stmt, {paramname: 'c'})
        eq_(res, 3)

    @tough_parameters
    def test_standalone_bindparam_escape_expanding(self, paramname, connection, multirow_fixture):
        tbl1 = multirow_fixture
        stmt = select(tbl1.c.myid).where(tbl1.c.name.in_(bindparam(paramname, value=['a', 'b']))).order_by(tbl1.c.myid)
        res = connection.scalars(stmt, {paramname: ['d', 'a']}).all()
        eq_(res, [1, 4])