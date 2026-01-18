from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Set
from sqlalchemy import CHAR
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import event
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Numeric
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from ... import autogenerate
from ... import util
from ...autogenerate import api
from ...ddl.base import _fk_spec
from ...migration import MigrationContext
from ...operations import ops
from ...testing import config
from ...testing import eq_
from ...testing.env import clear_staging_env
from ...testing.env import staging_env
def _assert_fk_diff(self, diff, type_, source_table, source_columns, target_table, target_columns, name=None, conditional_name=None, source_schema=None, onupdate=None, ondelete=None, initially=None, deferrable=None):
    fk_source_schema, fk_source_table, fk_source_columns, fk_target_schema, fk_target_table, fk_target_columns, fk_onupdate, fk_ondelete, fk_deferrable, fk_initially = _fk_spec(diff[1])
    eq_(diff[0], type_)
    eq_(fk_source_table, source_table)
    eq_(fk_source_columns, source_columns)
    eq_(fk_target_table, target_table)
    eq_(fk_source_schema, source_schema)
    eq_(fk_onupdate, onupdate)
    eq_(fk_ondelete, ondelete)
    eq_(fk_initially, initially)
    eq_(fk_deferrable, deferrable)
    eq_([elem.column.name for elem in diff[1].elements], target_columns)
    if conditional_name is not None:
        if conditional_name == 'servergenerated':
            fks = inspect(self.bind).get_foreign_keys(source_table)
            server_fk_name = fks[0]['name']
            eq_(diff[1].name, server_fk_name)
        else:
            eq_(diff[1].name, conditional_name)
    else:
        eq_(diff[1].name, name)