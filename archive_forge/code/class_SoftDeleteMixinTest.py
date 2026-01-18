from collections import abc
import datetime
from unittest import mock
from sqlalchemy import Column
from sqlalchemy import Integer, String
from sqlalchemy import event
from sqlalchemy.orm import declarative_base
from oslo_db.sqlalchemy import models
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class SoftDeleteMixinTest(db_test_base._DbTestCase):

    def setUp(self):
        super(SoftDeleteMixinTest, self).setUp()
        t = BASE.metadata.tables['test_model_soft_deletes']
        t.create(self.engine)
        self.addCleanup(t.drop, self.engine)
        self.session = self.sessionmaker(autocommit=False)
        self.addCleanup(self.session.close)

    @mock.patch('oslo_utils.timeutils.utcnow')
    def test_soft_delete(self, mock_utcnow):
        dt = datetime.datetime.utcnow().replace(microsecond=0)
        mock_utcnow.return_value = dt
        m = SoftDeletedModel(id=123456, smth='test')
        self.session.add(m)
        self.session.commit()
        self.assertEqual(0, m.deleted)
        self.assertIsNone(m.deleted_at)
        m.soft_delete(self.session)
        self.assertEqual(123456, m.deleted)
        self.assertIs(dt, m.deleted_at)

    def test_soft_delete_coerce_deleted_to_integer(self):

        def listener(conn, cur, stmt, params, context, executemany):
            if 'insert' in stmt.lower():
                self.assertNotIn('False', str(params))
        event.listen(self.engine, 'before_cursor_execute', listener)
        self.addCleanup(event.remove, self.engine, 'before_cursor_execute', listener)
        m = SoftDeletedModel(id=1, smth='test', deleted=False)
        self.session.add(m)
        self.session.commit()

    def test_deleted_set_to_null(self):
        m = SoftDeletedModel(id=123456, smth='test')
        self.session.add(m)
        self.session.commit()
        m.deleted = None
        self.session.commit()
        self.assertIsNone(m.deleted)