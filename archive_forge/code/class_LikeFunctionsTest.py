import collections.abc as collections_abc
import itertools
from .. import AssertsCompiledSQL
from .. import AssertsExecutionResults
from .. import config
from .. import fixtures
from ..assertions import assert_raises
from ..assertions import eq_
from ..assertions import in_
from ..assertsql import CursorSQL
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import case
from ... import column
from ... import Computed
from ... import exists
from ... import false
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import null
from ... import select
from ... import String
from ... import table
from ... import testing
from ... import text
from ... import true
from ... import tuple_
from ... import TupleType
from ... import union
from ... import values
from ...exc import DatabaseError
from ...exc import ProgrammingError
class LikeFunctionsTest(fixtures.TablesTest):
    __backend__ = True
    run_inserts = 'once'
    run_deletes = None

    @classmethod
    def define_tables(cls, metadata):
        Table('some_table', metadata, Column('id', Integer, primary_key=True), Column('data', String(50)))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.some_table.insert(), [{'id': 1, 'data': 'abcdefg'}, {'id': 2, 'data': 'ab/cdefg'}, {'id': 3, 'data': 'ab%cdefg'}, {'id': 4, 'data': 'ab_cdefg'}, {'id': 5, 'data': 'abcde/fg'}, {'id': 6, 'data': 'abcde%fg'}, {'id': 7, 'data': 'ab#cdefg'}, {'id': 8, 'data': 'ab9cdefg'}, {'id': 9, 'data': 'abcde#fg'}, {'id': 10, 'data': 'abcd9fg'}, {'id': 11, 'data': None}])

    def _test(self, expr, expected):
        some_table = self.tables.some_table
        with config.db.connect() as conn:
            rows = {value for value, in conn.execute(select(some_table.c.id).where(expr))}
        eq_(rows, expected)

    def test_startswith_unescaped(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith('ab%c'), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})

    def test_startswith_autoescape(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith('ab%c', autoescape=True), {3})

    def test_startswith_sqlexpr(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith(literal_column("'ab%c'")), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})

    def test_startswith_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith('ab##c', escape='#'), {7})

    def test_startswith_autoescape_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith('ab%c', autoescape=True, escape='#'), {3})
        self._test(col.startswith('ab#c', autoescape=True, escape='#'), {7})

    def test_endswith_unescaped(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith('e%fg'), {1, 2, 3, 4, 5, 6, 7, 8, 9})

    def test_endswith_sqlexpr(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith(literal_column("'e%fg'")), {1, 2, 3, 4, 5, 6, 7, 8, 9})

    def test_endswith_autoescape(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith('e%fg', autoescape=True), {6})

    def test_endswith_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith('e##fg', escape='#'), {9})

    def test_endswith_autoescape_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith('e%fg', autoescape=True, escape='#'), {6})
        self._test(col.endswith('e#fg', autoescape=True, escape='#'), {9})

    def test_contains_unescaped(self):
        col = self.tables.some_table.c.data
        self._test(col.contains('b%cde'), {1, 2, 3, 4, 5, 6, 7, 8, 9})

    def test_contains_autoescape(self):
        col = self.tables.some_table.c.data
        self._test(col.contains('b%cde', autoescape=True), {3})

    def test_contains_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.contains('b##cde', escape='#'), {7})

    def test_contains_autoescape_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.contains('b%cd', autoescape=True, escape='#'), {3})
        self._test(col.contains('b#cd', autoescape=True, escape='#'), {7})

    @testing.requires.regexp_match
    def test_not_regexp_match(self):
        col = self.tables.some_table.c.data
        self._test(~col.regexp_match('a.cde'), {2, 3, 4, 7, 8, 10})

    @testing.requires.regexp_replace
    def test_regexp_replace(self):
        col = self.tables.some_table.c.data
        self._test(col.regexp_replace('a.cde', 'FOO').contains('FOO'), {1, 5, 6, 9})

    @testing.requires.regexp_match
    @testing.combinations(('a.cde', {1, 5, 6, 9}), ('abc', {1, 5, 6, 9, 10}), ('^abc', {1, 5, 6, 9, 10}), ('9cde', {8}), ('^a', set(range(1, 11))), ('(b|c)', set(range(1, 11))), ('^(b|c)', set()))
    def test_regexp_match(self, text, expected):
        col = self.tables.some_table.c.data
        self._test(col.regexp_match(text), expected)