from .. import config
from .. import fixtures
from ..assertions import eq_
from ..assertions import is_true
from ..config import requirements
from ..provision import normalize_sequence
from ..schema import Column
from ..schema import Table
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import Sequence
from ... import String
from ... import testing
class SequenceCompilerTest(testing.AssertsCompiledSQL, fixtures.TestBase):
    __requires__ = ('sequences',)
    __backend__ = True

    def test_literal_binds_inline_compile(self, connection):
        table = Table('x', MetaData(), Column('y', Integer, normalize_sequence(config, Sequence('y_seq'))), Column('q', Integer))
        stmt = table.insert().values(q=5)
        seq_nextval = connection.dialect.statement_compiler(statement=None, dialect=connection.dialect).visit_sequence(normalize_sequence(config, Sequence('y_seq')))
        self.assert_compile(stmt, 'INSERT INTO x (y, q) VALUES (%s, 5)' % (seq_nextval,), literal_binds=True, dialect=connection.dialect)