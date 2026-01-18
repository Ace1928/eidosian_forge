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
class DBAPIResourceDataTest(common.HeatTestCase):

    def setUp(self):
        super(DBAPIResourceDataTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.template = create_raw_template(self.ctx)
        self.user_creds = create_user_creds(self.ctx)
        self.stack = create_stack(self.ctx, self.template, self.user_creds)
        self.resource = create_resource(self.ctx, self.stack)
        self.resource.context = self.ctx

    def test_resource_data_set_get(self):
        create_resource_data(self.ctx, self.resource)
        val = db_api.resource_data_get(self.ctx, self.resource.id, 'test_resource_key')
        self.assertEqual('test_value', val)
        create_resource_data(self.ctx, self.resource, value='foo')
        val = db_api.resource_data_get(self.ctx, self.resource.id, 'test_resource_key')
        self.assertEqual('foo', val)
        create_resource_data(self.ctx, self.resource, key='encryped_resource_key', redact=True)
        val = db_api.resource_data_get(self.ctx, self.resource.id, 'encryped_resource_key')
        self.assertEqual('test_value', val)
        vals = db_api.resource_data_get_all(self.resource.context, self.resource.id)
        self.assertEqual(2, len(vals))
        self.assertEqual('foo', vals.get('test_resource_key'))
        self.assertEqual('test_value', vals.get('encryped_resource_key'))
        self.resource = db_api.resource_get(self.ctx, self.resource.id)
        vals = db_api.resource_data_get_all(self.ctx, None, self.resource.data)
        self.assertEqual(2, len(vals))
        self.assertEqual('foo', vals.get('test_resource_key'))
        self.assertEqual('test_value', vals.get('encryped_resource_key'))

    def test_resource_data_delete(self):
        create_resource_data(self.ctx, self.resource)
        res_data = db_api.resource_data_get_by_key(self.ctx, self.resource.id, 'test_resource_key')
        self.assertIsNotNone(res_data)
        self.assertEqual('test_value', res_data.value)
        db_api.resource_data_delete(self.ctx, self.resource.id, 'test_resource_key')
        self.assertRaises(exception.NotFound, db_api.resource_data_get_by_key, self.ctx, self.resource.id, 'test_resource_key')
        self.assertIsNotNone(res_data)
        self.assertRaises(exception.NotFound, db_api.resource_data_get_all, self.resource.context, self.resource.id)