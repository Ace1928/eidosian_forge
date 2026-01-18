import operator
import re
import sqlalchemy as sa
from .. import config
from .. import engines
from .. import eq_
from .. import expect_raises
from .. import expect_raises_message
from .. import expect_warnings
from .. import fixtures
from .. import is_
from ..provision import get_temp_table_name
from ..provision import temp_table_keyword_args
from ..schema import Column
from ..schema import Table
from ... import event
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import String
from ... import testing
from ... import types as sql_types
from ...engine import Inspector
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...exc import NoSuchTableError
from ...exc import UnreflectableTableError
from ...schema import DDL
from ...schema import Index
from ...sql.elements import quoted_name
from ...sql.schema import BLANK_SCHEMA
from ...testing import ComparesIndexes
from ...testing import ComparesTables
from ...testing import is_false
from ...testing import is_true
from ...testing import mock
def _test_get_comments(self, connection, schema=None):
    insp = inspect(connection)
    exp = self.exp_comments(schema=schema)
    eq_(insp.get_table_comment('comment_test', schema=schema), exp[schema, 'comment_test'])
    eq_(insp.get_table_comment('users', schema=schema), exp[schema, 'users'])
    eq_(insp.get_table_comment('comment_test', schema=schema), exp[schema, 'comment_test'])
    no_cst = self.tables.no_constraints.name
    eq_(insp.get_table_comment(no_cst, schema=schema), exp[schema, no_cst])