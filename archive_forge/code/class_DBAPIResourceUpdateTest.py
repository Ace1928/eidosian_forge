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
class DBAPIResourceUpdateTest(common.HeatTestCase):

    def setUp(self):
        super(DBAPIResourceUpdateTest, self).setUp()
        self.ctx = utils.dummy_context()
        template = create_raw_template(self.ctx)
        user_creds = create_user_creds(self.ctx)
        stack = create_stack(self.ctx, template, user_creds)
        self.resource = create_resource(self.ctx, stack, False, atomic_key=0)

    def test_unlocked_resource_update(self):
        values = {'engine_id': 'engine-1', 'action': 'CREATE', 'status': 'IN_PROGRESS'}
        db_res = db_api.resource_get(self.ctx, self.resource.id)
        ret = db_api.resource_update(self.ctx, self.resource.id, values, db_res.atomic_key, None)
        self.assertTrue(ret)
        db_res = db_api.resource_get(self.ctx, self.resource.id, refresh=True)
        self.assertEqual('engine-1', db_res.engine_id)
        self.assertEqual('CREATE', db_res.action)
        self.assertEqual('IN_PROGRESS', db_res.status)
        self.assertEqual(1, db_res.atomic_key)

    def test_locked_resource_update_by_same_engine(self):
        values = {'engine_id': 'engine-1', 'action': 'CREATE', 'status': 'IN_PROGRESS'}
        db_res = db_api.resource_get(self.ctx, self.resource.id)
        ret = db_api.resource_update(self.ctx, self.resource.id, values, db_res.atomic_key, None)
        self.assertTrue(ret)
        db_res = db_api.resource_get(self.ctx, self.resource.id, refresh=True)
        self.assertEqual('engine-1', db_res.engine_id)
        self.assertEqual(1, db_res.atomic_key)
        values = {'engine_id': 'engine-1', 'action': 'CREATE', 'status': 'FAILED'}
        ret = db_api.resource_update(self.ctx, self.resource.id, values, db_res.atomic_key, 'engine-1')
        self.assertTrue(ret)
        db_res = db_api.resource_get(self.ctx, self.resource.id, refresh=True)
        self.assertEqual('engine-1', db_res.engine_id)
        self.assertEqual('CREATE', db_res.action)
        self.assertEqual('FAILED', db_res.status)
        self.assertEqual(2, db_res.atomic_key)

    def test_locked_resource_update_by_other_engine(self):
        values = {'engine_id': 'engine-1', 'action': 'CREATE', 'status': 'IN_PROGRESS'}
        db_res = db_api.resource_get(self.ctx, self.resource.id)
        ret = db_api.resource_update(self.ctx, self.resource.id, values, db_res.atomic_key, None)
        self.assertTrue(ret)
        db_res = db_api.resource_get(self.ctx, self.resource.id, refresh=True)
        self.assertEqual('engine-1', db_res.engine_id)
        self.assertEqual(1, db_res.atomic_key)
        values = {'engine_id': 'engine-2', 'action': 'CREATE', 'status': 'FAILED'}
        ret = db_api.resource_update(self.ctx, self.resource.id, values, db_res.atomic_key, 'engine-2')
        self.assertFalse(ret)

    def test_release_resource_lock(self):
        values = {'engine_id': 'engine-1', 'action': 'CREATE', 'status': 'IN_PROGRESS'}
        db_res = db_api.resource_get(self.ctx, self.resource.id)
        ret = db_api.resource_update(self.ctx, self.resource.id, values, db_res.atomic_key, None)
        self.assertTrue(ret)
        db_res = db_api.resource_get(self.ctx, self.resource.id, refresh=True)
        self.assertEqual('engine-1', db_res.engine_id)
        self.assertEqual(1, db_res.atomic_key)
        values = {'engine_id': None, 'action': 'CREATE', 'status': 'COMPLETE'}
        ret = db_api.resource_update(self.ctx, self.resource.id, values, db_res.atomic_key, 'engine-1')
        self.assertTrue(ret)
        db_res = db_api.resource_get(self.ctx, self.resource.id, refresh=True)
        self.assertIsNone(db_res.engine_id)
        self.assertEqual('CREATE', db_res.action)
        self.assertEqual('COMPLETE', db_res.status)
        self.assertEqual(2, db_res.atomic_key)

    def test_steal_resource_lock(self):
        values = {'engine_id': 'engine-1', 'action': 'CREATE', 'status': 'IN_PROGRESS'}
        db_res = db_api.resource_get(self.ctx, self.resource.id)
        ret = db_api.resource_update(self.ctx, self.resource.id, values, db_res.atomic_key, None)
        self.assertTrue(ret)
        db_res = db_api.resource_get(self.ctx, self.resource.id, refresh=True)
        self.assertEqual('engine-1', db_res.engine_id)
        self.assertEqual(1, db_res.atomic_key)
        values = {'engine_id': 'engine-2', 'action': 'DELETE', 'status': 'IN_PROGRESS'}
        ret = db_api.resource_update(self.ctx, self.resource.id, values, db_res.atomic_key, 'engine-1')
        self.assertTrue(ret)
        db_res = db_api.resource_get(self.ctx, self.resource.id, refresh=True)
        self.assertEqual('engine-2', db_res.engine_id)
        self.assertEqual('DELETE', db_res.action)
        self.assertEqual(2, db_res.atomic_key)