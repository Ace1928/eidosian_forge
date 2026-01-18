import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_log import log
import sqlalchemy
from sqlalchemy import exc
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import sql
from keystone.common.sql import core
import keystone.conf
from keystone.credential.providers import fernet as credential_provider
from keystone import exception
from keystone.identity.backends import sql_model as identity_sql
from keystone.resource.backends import base as resource
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.limit import test_backends as limit_tests
from keystone.tests.unit.policy import test_backends as policy_tests
from keystone.tests.unit.resource import test_backends as resource_tests
from keystone.tests.unit.trust import test_backends as trust_tests
from keystone.trust.backends import sql as trust_sql
class SqlModels(SqlTests):

    def load_table(self, name):
        table = sqlalchemy.Table(name, sql.ModelBase.metadata, autoload_with=self.database_fixture.engine)
        return table

    def assertExpectedSchema(self, table, expected_schema):
        """Assert that a table's schema is what we expect.

        :param string table: the name of the table to inspect
        :param tuple expected_schema: a tuple of tuples containing the
            expected schema
        :raises AssertionError: when the database schema doesn't match the
            expected schema

        The expected_schema format is simply::

            (
                ('column name', sql type, qualifying detail),
                ...
            )

        The qualifying detail varies based on the type of the column::

          - sql.Boolean columns must indicate the column's default value or
            None if there is no default
          - Columns with a length, like sql.String, must indicate the
            column's length
          - All other column types should use None

        Example::

            cols = (('id', sql.String, 64),
                    ('enabled', sql.Boolean, True),
                    ('extra', sql.JsonBlob, None))
            self.assertExpectedSchema('table_name', cols)

        """
        table = self.load_table(table)
        actual_schema = []
        for column in table.c:
            if isinstance(column.type, sql.Boolean):
                default = None
                if column.default:
                    default = column.default.arg
                actual_schema.append((column.name, type(column.type), default))
            elif hasattr(column.type, 'length') and (not isinstance(column.type, sql.Enum)):
                actual_schema.append((column.name, type(column.type), column.type.length))
            else:
                actual_schema.append((column.name, type(column.type), None))
        self.assertCountEqual(expected_schema, actual_schema)

    def test_user_model(self):
        cols = (('id', sql.String, 64), ('domain_id', sql.String, 64), ('default_project_id', sql.String, 64), ('enabled', sql.Boolean, None), ('extra', sql.JsonBlob, None), ('created_at', sql.DateTime, None), ('last_active_at', sqlalchemy.Date, None))
        self.assertExpectedSchema('user', cols)

    def test_local_user_model(self):
        cols = (('id', sql.Integer, None), ('user_id', sql.String, 64), ('name', sql.String, 255), ('domain_id', sql.String, 64), ('failed_auth_count', sql.Integer, None), ('failed_auth_at', sql.DateTime, None))
        self.assertExpectedSchema('local_user', cols)

    def test_password_model(self):
        cols = (('id', sql.Integer, None), ('local_user_id', sql.Integer, None), ('password_hash', sql.String, 255), ('created_at', sql.DateTime, None), ('expires_at', sql.DateTime, None), ('created_at_int', sql.DateTimeInt, None), ('expires_at_int', sql.DateTimeInt, None), ('self_service', sql.Boolean, False))
        self.assertExpectedSchema('password', cols)

    def test_federated_user_model(self):
        cols = (('id', sql.Integer, None), ('user_id', sql.String, 64), ('idp_id', sql.String, 64), ('protocol_id', sql.String, 64), ('unique_id', sql.String, 255), ('display_name', sql.String, 255))
        self.assertExpectedSchema('federated_user', cols)

    def test_nonlocal_user_model(self):
        cols = (('domain_id', sql.String, 64), ('name', sql.String, 255), ('user_id', sql.String, 64))
        self.assertExpectedSchema('nonlocal_user', cols)

    def test_group_model(self):
        cols = (('id', sql.String, 64), ('name', sql.String, 64), ('description', sql.Text, None), ('domain_id', sql.String, 64), ('extra', sql.JsonBlob, None))
        self.assertExpectedSchema('group', cols)

    def test_project_model(self):
        cols = (('id', sql.String, 64), ('name', sql.String, 64), ('description', sql.Text, None), ('domain_id', sql.String, 64), ('enabled', sql.Boolean, None), ('extra', sql.JsonBlob, None), ('parent_id', sql.String, 64), ('is_domain', sql.Boolean, False))
        self.assertExpectedSchema('project', cols)

    def test_role_assignment_model(self):
        cols = (('type', sql.Enum, None), ('actor_id', sql.String, 64), ('target_id', sql.String, 64), ('role_id', sql.String, 64), ('inherited', sql.Boolean, False))
        self.assertExpectedSchema('assignment', cols)

    def test_user_group_membership(self):
        cols = (('group_id', sql.String, 64), ('user_id', sql.String, 64))
        self.assertExpectedSchema('user_group_membership', cols)

    def test_revocation_event_model(self):
        cols = (('id', sql.Integer, None), ('domain_id', sql.String, 64), ('project_id', sql.String, 64), ('user_id', sql.String, 64), ('role_id', sql.String, 64), ('trust_id', sql.String, 64), ('consumer_id', sql.String, 64), ('access_token_id', sql.String, 64), ('issued_before', sql.DateTime, None), ('expires_at', sql.DateTime, None), ('revoked_at', sql.DateTime, None), ('audit_id', sql.String, 32), ('audit_chain_id', sql.String, 32))
        self.assertExpectedSchema('revocation_event', cols)

    def test_project_tags_model(self):
        cols = (('project_id', sql.String, 64), ('name', sql.Unicode, 255))
        self.assertExpectedSchema('project_tag', cols)