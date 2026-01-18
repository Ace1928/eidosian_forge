from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def _assert_row(self, pk, values):
    row = self.session.execute(sql.select(MyModel.__table__).where(MyModel.__table__.c.id == pk)).first()
    values['id'] = pk
    self.assertEqual(values, dict(row._mapping))