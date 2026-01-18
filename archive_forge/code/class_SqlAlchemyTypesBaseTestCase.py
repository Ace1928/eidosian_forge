import abc
import netaddr
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_utils import timeutils
from oslo_utils import uuidutils
import sqlalchemy as sa
from neutron_lib import context
from neutron_lib.db import sqlalchemytypes
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
class SqlAlchemyTypesBaseTestCase(test_fixtures.OpportunisticDBTestMixin, test_base.BaseTestCase, metaclass=abc.ABCMeta):

    def setUp(self):
        super().setUp()
        self.engine = enginefacade.writer.get_engine()
        meta = sa.MetaData()
        self.test_table = self._get_test_table(meta)
        meta.create_all(self.engine)
        self.addCleanup(meta.drop_all, self.engine)
        self.ctxt = context.get_admin_context()

    @abc.abstractmethod
    def _get_test_table(self, meta):
        """Returns a new sa.Table() object for this test case."""

    def _add_row(self, **kargs):
        row_insert = self.test_table.insert().values(**kargs)
        with self.engine.connect() as conn, conn.begin():
            conn.execute(row_insert)

    def _get_all(self):
        rows_select = self.test_table.select()
        with self.engine.connect() as conn, conn.begin():
            return conn.execute(rows_select).fetchall()

    def _update_row(self, **kargs):
        row_update = self.test_table.update().values(**kargs)
        with self.engine.connect() as conn, conn.begin():
            conn.execute(row_update)

    def _delete_rows(self):
        row_delete = self.test_table.delete()
        with self.engine.connect() as conn, conn.begin():
            conn.execute(row_delete)