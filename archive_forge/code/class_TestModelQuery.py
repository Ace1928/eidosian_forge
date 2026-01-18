from unittest import mock
from urllib import parse
import fixtures
import sqlalchemy
from sqlalchemy import Boolean, Index, Integer, DateTime, String
from sqlalchemy import MetaData, Table, Column
from sqlalchemy import ForeignKey, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import psycopg2
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import column_property
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import sql
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql import select
from sqlalchemy.types import UserDefinedType
from oslo_db import exception
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class TestModelQuery(test_base.BaseTestCase):

    def setUp(self):
        super(TestModelQuery, self).setUp()
        self.session = mock.MagicMock()
        self.session.query.return_value = self.session.query
        self.session.query.filter.return_value = self.session.query

    def test_wrong_model(self):
        self.assertRaises(TypeError, utils.model_query, FakeModel, session=self.session)

    def test_no_soft_deleted(self):
        self.assertRaises(ValueError, utils.model_query, MyModel, session=self.session, deleted=True)

    def test_deleted_false(self):
        mock_query = utils.model_query(MyModelSoftDeleted, session=self.session, deleted=False)
        deleted_filter = mock_query.filter.call_args[0][0]
        self.assertEqual('soft_deleted_test_model.deleted = :deleted_1', str(deleted_filter))
        self.assertEqual(deleted_filter.right.value, MyModelSoftDeleted.__mapper__.c.deleted.default.arg)

    def test_deleted_true(self):
        mock_query = utils.model_query(MyModelSoftDeleted, session=self.session, deleted=True)
        deleted_filter = mock_query.filter.call_args[0][0]
        self.assertEqual(str(deleted_filter), 'soft_deleted_test_model.deleted != :deleted_1')
        self.assertEqual(deleted_filter.right.value, MyModelSoftDeleted.__mapper__.c.deleted.default.arg)

    @mock.patch.object(utils, '_read_deleted_filter')
    def test_no_deleted_value(self, _read_deleted_filter):
        utils.model_query(MyModelSoftDeleted, session=self.session)
        self.assertEqual(0, _read_deleted_filter.call_count)

    def test_project_filter(self):
        project_id = 10
        mock_query = utils.model_query(MyModelSoftDeletedProjectId, session=self.session, project_only=True, project_id=project_id)
        deleted_filter = mock_query.filter.call_args[0][0]
        self.assertEqual('soft_deleted_project_id_test_model.project_id = :project_id_1', str(deleted_filter))
        self.assertEqual(project_id, deleted_filter.right.value)

    def test_project_filter_wrong_model(self):
        self.assertRaises(ValueError, utils.model_query, MyModelSoftDeleted, session=self.session, project_id=10)

    def test_project_filter_allow_none(self):
        mock_query = utils.model_query(MyModelSoftDeletedProjectId, session=self.session, project_id=(10, None))
        self.assertEqual('soft_deleted_project_id_test_model.project_id IN (:project_id_1, NULL)', str(mock_query.filter.call_args[0][0]))

    def test_model_query_common(self):
        utils.model_query(MyModel, args=(MyModel.id,), session=self.session)
        self.session.query.assert_called_with(MyModel.id)