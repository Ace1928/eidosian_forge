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
class ExceptionTest(fixtures.TablesTest):
    """Test basic exception wrapping.

    DBAPIs vary a lot in exception behavior so to actually anticipate
    specific exceptions from real round trips, we need to be conservative.

    """
    run_deletes = 'each'
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('manual_pk', metadata, Column('id', Integer, primary_key=True, autoincrement=False), Column('data', String(50)))

    @requirements.duplicate_key_raises_integrity_error
    def test_integrity_error(self):
        with config.db.connect() as conn:
            trans = conn.begin()
            conn.execute(self.tables.manual_pk.insert(), {'id': 1, 'data': 'd1'})
            assert_raises(exc.IntegrityError, conn.execute, self.tables.manual_pk.insert(), {'id': 1, 'data': 'd1'})
            trans.rollback()

    def test_exception_with_non_ascii(self):
        with config.db.connect() as conn:
            try:
                conn.execute(select(literal_column('méil')))
                assert False
            except exc.DBAPIError as err:
                err_str = str(err)
                assert str(err.orig) in str(err)
            assert isinstance(err_str, str)