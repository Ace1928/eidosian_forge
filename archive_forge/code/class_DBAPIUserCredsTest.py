import datetime
import json
import time
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_utils import timeutils
from sqlalchemy import orm
from sqlalchemy.orm import exc
from sqlalchemy.orm import session
from heat.common import context
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import resource as rsrc
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.engine import template_files
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class DBAPIUserCredsTest(common.HeatTestCase):

    def setUp(self):
        super(DBAPIUserCredsTest, self).setUp()
        self.ctx = utils.dummy_context()

    def test_user_creds_create_trust(self):
        user_creds = create_user_creds(self.ctx, trust_id='test_trust_id', trustor_user_id='trustor_id')
        self.assertIsNotNone(user_creds['id'])
        self.assertEqual('test_trust_id', user_creds['trust_id'])
        self.assertEqual('trustor_id', user_creds['trustor_user_id'])
        self.assertIsNone(user_creds['username'])
        self.assertIsNone(user_creds['password'])
        self.assertEqual(self.ctx.project_name, user_creds['tenant'])
        self.assertEqual(self.ctx.tenant_id, user_creds['tenant_id'])

    def test_user_creds_create_password(self):
        user_creds = create_user_creds(self.ctx)
        self.assertIsNotNone(user_creds['id'])
        self.assertEqual(self.ctx.password, user_creds['password'])

    def test_user_creds_get(self):
        user_creds = create_user_creds(self.ctx)
        ret_user_creds = db_api.user_creds_get(self.ctx, user_creds['id'])
        self.assertEqual(user_creds['password'], ret_user_creds['password'])

    def test_user_creds_get_noexist(self):
        self.assertIsNone(db_api.user_creds_get(self.ctx, 123456))

    def test_user_creds_delete(self):
        user_creds = create_user_creds(self.ctx)
        self.assertIsNotNone(user_creds['id'])
        db_api.user_creds_delete(self.ctx, user_creds['id'])
        creds = db_api.user_creds_get(self.ctx, user_creds['id'])
        self.assertIsNone(creds)
        mock_delete = self.patchobject(session.Session, 'delete')
        err = self.assertRaises(exception.NotFound, db_api.user_creds_delete, self.ctx, user_creds['id'])
        exp_msg = 'Attempt to delete user creds with id %s that does not exist' % user_creds['id']
        self.assertIn(exp_msg, str(err))
        self.assertEqual(0, mock_delete.call_count)

    def test_user_creds_delete_retries(self):
        mock_delete = self.patchobject(session.Session, 'delete')
        mock_delete.side_effect = [exc.StaleDataError, exc.StaleDataError, None]
        user_creds = create_user_creds(self.ctx)
        self.assertIsNotNone(user_creds['id'])
        self.assertIsNone(db_api.user_creds_delete(self.ctx, user_creds['id']))
        self.assertEqual(3, mock_delete.call_count)
        mock_delete.side_effect = [exc.UnmappedError]
        self.assertRaises(exc.UnmappedError, db_api.user_creds_delete, self.ctx, user_creds['id'])
        self.assertEqual(4, mock_delete.call_count)