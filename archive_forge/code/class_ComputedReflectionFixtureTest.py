from __future__ import annotations
import itertools
import random
import re
import sys
import sqlalchemy as sa
from .base import TestBase
from .. import config
from .. import mock
from ..assertions import eq_
from ..assertions import ne_
from ..util import adict
from ..util import drop_all_tables_from_metadata
from ... import event
from ... import util
from ...schema import sort_tables_and_constraints
from ...sql import visitors
from ...sql.elements import ClauseElement
class ComputedReflectionFixtureTest(TablesTest):
    run_inserts = run_deletes = None
    __backend__ = True
    __requires__ = ('computed_columns', 'table_reflection')
    regexp = re.compile('[\\[\\]\\(\\)\\s`\'\\"]*')

    def normalize(self, text):
        return self.regexp.sub('', text).lower()

    @classmethod
    def define_tables(cls, metadata):
        from ... import Integer
        from ... import testing
        from ...schema import Column
        from ...schema import Computed
        from ...schema import Table
        Table('computed_default_table', metadata, Column('id', Integer, primary_key=True), Column('normal', Integer), Column('computed_col', Integer, Computed('normal + 42')), Column('with_default', Integer, server_default='42'))
        t = Table('computed_column_table', metadata, Column('id', Integer, primary_key=True), Column('normal', Integer), Column('computed_no_flag', Integer, Computed('normal + 42')))
        if testing.requires.schemas.enabled:
            t2 = Table('computed_column_table', metadata, Column('id', Integer, primary_key=True), Column('normal', Integer), Column('computed_no_flag', Integer, Computed('normal / 42')), schema=config.test_schema)
        if testing.requires.computed_columns_virtual.enabled:
            t.append_column(Column('computed_virtual', Integer, Computed('normal + 2', persisted=False)))
            if testing.requires.schemas.enabled:
                t2.append_column(Column('computed_virtual', Integer, Computed('normal / 2', persisted=False)))
        if testing.requires.computed_columns_stored.enabled:
            t.append_column(Column('computed_stored', Integer, Computed('normal - 42', persisted=True)))
            if testing.requires.schemas.enabled:
                t2.append_column(Column('computed_stored', Integer, Computed('normal * 42', persisted=True)))