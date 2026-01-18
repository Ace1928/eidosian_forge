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
class DBAPIResourceTest(common.HeatTestCase):

    def setUp(self):
        super(DBAPIResourceTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.template = create_raw_template(self.ctx)
        self.user_creds = create_user_creds(self.ctx)
        self.stack = create_stack(self.ctx, self.template, self.user_creds)

    def test_resource_create(self):
        res = create_resource(self.ctx, self.stack)
        ret_res = db_api.resource_get(self.ctx, res.id)
        self.assertIsNotNone(ret_res)
        self.assertEqual('test_resource_name', ret_res.name)
        self.assertEqual(UUID1, ret_res.physical_resource_id)
        self.assertEqual('create', ret_res.action)
        self.assertEqual('complete', ret_res.status)
        self.assertEqual('create_complete', ret_res.status_reason)
        self.assertEqual('{"foo": "123"}', json.dumps(ret_res.rsrc_metadata))
        self.assertEqual(self.stack.id, ret_res.stack_id)

    def test_resource_get(self):
        res = create_resource(self.ctx, self.stack)
        ret_res = db_api.resource_get(self.ctx, res.id)
        self.assertIsNotNone(ret_res)
        self.assertRaises(exception.NotFound, db_api.resource_get, self.ctx, UUID2)

    def test_resource_get_by_name_and_stack(self):
        create_resource(self.ctx, self.stack)
        ret_res = db_api.resource_get_by_name_and_stack(self.ctx, 'test_resource_name', self.stack.id)
        self.assertIsNotNone(ret_res)
        self.assertEqual('test_resource_name', ret_res.name)
        self.assertEqual(self.stack.id, ret_res.stack_id)
        self.assertIsNone(db_api.resource_get_by_name_and_stack(self.ctx, 'abc', self.stack.id))

    def test_resource_get_by_physical_resource_id(self):
        create_resource(self.ctx, self.stack)
        ret_res = db_api.resource_get_by_physical_resource_id(self.ctx, UUID1)
        self.assertIsNotNone(ret_res)
        self.assertEqual(UUID1, ret_res.physical_resource_id)
        self.assertIsNone(db_api.resource_get_by_physical_resource_id(self.ctx, UUID2))

    def test_resource_get_all_by_physical_resource_id(self):
        create_resource(self.ctx, self.stack)
        create_resource(self.ctx, self.stack)
        ret_res = db_api.resource_get_all_by_physical_resource_id(self.ctx, UUID1)
        ret_list = list(ret_res)
        self.assertEqual(2, len(ret_list))
        for res in ret_list:
            self.assertEqual(UUID1, res.physical_resource_id)
        mt = db_api.resource_get_all_by_physical_resource_id(self.ctx, UUID2)
        self.assertFalse(list(mt))

    def test_resource_get_all_by_with_admin_context(self):
        admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
        create_resource(self.ctx, self.stack, phys_res_id=UUID1)
        create_resource(self.ctx, self.stack, phys_res_id=UUID2)
        ret_res = db_api.resource_get_all_by_physical_resource_id(admin_ctx, UUID1)
        ret_list = list(ret_res)
        self.assertEqual(1, len(ret_list))
        self.assertEqual(UUID1, ret_list[0].physical_resource_id)
        mt = db_api.resource_get_all_by_physical_resource_id(admin_ctx, UUID2)
        ret_list = list(mt)
        self.assertEqual(1, len(ret_list))
        self.assertEqual(UUID2, ret_list[0].physical_resource_id)

    def test_resource_get_all(self):
        values = [{'name': 'res1'}, {'name': 'res2'}, {'name': 'res3'}]
        [create_resource(self.ctx, self.stack, False, **val) for val in values]
        resources = db_api.resource_get_all(self.ctx)
        self.assertEqual(3, len(resources))
        names = [resource.name for resource in resources]
        [self.assertIn(val['name'], names) for val in values]

    def test_resource_get_all_by_stack(self):
        stack1 = create_stack(self.ctx, self.template, self.user_creds)
        stack2 = create_stack(self.ctx, self.template, self.user_creds)
        values = [{'name': 'res1', 'stack_id': self.stack.id}, {'name': 'res2', 'stack_id': self.stack.id}, {'name': 'res3', 'stack_id': self.stack.id}, {'name': 'res4', 'stack_id': stack1.id}]
        [create_resource(self.ctx, self.stack, False, **val) for val in values]
        resources = db_api.resource_get_all_by_stack(self.ctx, self.stack.id)
        self.assertEqual(3, len(resources))
        self.assertEqual('res1', resources.get('res1').name)
        self.assertEqual('res2', resources.get('res2').name)
        self.assertEqual('res3', resources.get('res3').name)
        resources = db_api.resource_get_all_by_stack(self.ctx, self.stack.id, filters=dict(name='res1'))
        self.assertEqual(1, len(resources))
        self.assertEqual('res1', resources.get('res1').name)
        resources = db_api.resource_get_all_by_stack(self.ctx, self.stack.id, filters=dict(name=['res1', 'res2']))
        self.assertEqual(2, len(resources))
        self.assertEqual('res1', resources.get('res1').name)
        self.assertEqual('res2', resources.get('res2').name)
        self.assertEqual({}, db_api.resource_get_all_by_stack(self.ctx, stack2.id))

    def test_resource_get_all_active_by_stack(self):
        values = [{'name': 'res1', 'action': rsrc.Resource.DELETE, 'status': rsrc.Resource.COMPLETE}, {'name': 'res2', 'action': rsrc.Resource.DELETE, 'status': rsrc.Resource.IN_PROGRESS}, {'name': 'res3', 'action': rsrc.Resource.UPDATE, 'status': rsrc.Resource.IN_PROGRESS}, {'name': 'res4', 'action': rsrc.Resource.UPDATE, 'status': rsrc.Resource.COMPLETE}, {'name': 'res5', 'action': rsrc.Resource.INIT, 'status': rsrc.Resource.COMPLETE}, {'name': 'res6'}]
        [create_resource(self.ctx, self.stack, **val) for val in values]
        resources = db_api.resource_get_all_active_by_stack(self.ctx, self.stack.id)
        self.assertEqual(5, len(resources))
        for rsrc_id, res in resources.items():
            self.assertIn(res.name, ['res2', 'res3', 'res4', 'res5', 'res6'])

    def test_resource_get_all_by_root_stack(self):
        stack1 = create_stack(self.ctx, self.template, self.user_creds)
        stack2 = create_stack(self.ctx, self.template, self.user_creds)
        create_resource(self.ctx, self.stack, name='res1', root_stack_id=self.stack.id)
        create_resource(self.ctx, self.stack, name='res2', root_stack_id=self.stack.id)
        create_resource(self.ctx, self.stack, name='res3', root_stack_id=self.stack.id)
        create_resource(self.ctx, stack1, name='res4', root_stack_id=self.stack.id)
        resources = db_api.resource_get_all_by_root_stack(self.ctx, self.stack.id)
        self.assertEqual(4, len(resources))
        resource_names = [r.name for r in resources.values()]
        self.assertEqual(['res1', 'res2', 'res3', 'res4'], sorted(resource_names))
        resources = db_api.resource_get_all_by_root_stack(self.ctx, self.stack.id, filters=dict(name='res1'))
        self.assertEqual(1, len(resources))
        resource_names = [r.name for r in resources.values()]
        self.assertEqual(['res1'], resource_names)
        self.assertEqual(1, len(resources))
        resources = db_api.resource_get_all_by_root_stack(self.ctx, self.stack.id, filters=dict(name=['res1', 'res2']))
        self.assertEqual(2, len(resources))
        resource_names = [r.name for r in resources.values()]
        self.assertEqual(['res1', 'res2'], sorted(resource_names))
        self.assertEqual({}, db_api.resource_get_all_by_root_stack(self.ctx, stack2.id))

    def test_resource_purge_deleted_by_stack(self):
        val = {'name': 'res1', 'action': rsrc.Resource.DELETE, 'status': rsrc.Resource.COMPLETE}
        resource = create_resource(self.ctx, self.stack, **val)
        db_api.resource_purge_deleted(self.ctx, self.stack.id)
        self.assertRaises(exception.NotFound, db_api.resource_get, self.ctx, resource.id)

    @mock.patch.object(time, 'sleep')
    def test_resource_purge_deleted_by_stack_retry_on_deadlock(self, m_sleep):
        val = {'name': 'res1', 'action': rsrc.Resource.DELETE, 'status': rsrc.Resource.COMPLETE}
        create_resource(self.ctx, self.stack, **val)
        with mock.patch('sqlalchemy.orm.query.Query.delete', side_effect=db_exception.DBDeadlock) as mock_delete:
            self.assertRaises(db_exception.DBDeadlock, db_api.resource_purge_deleted, self.ctx, self.stack.id)
            self.assertEqual(21, mock_delete.call_count)

    def test_engine_get_all_locked_by_stack(self):
        values = [{'name': 'res1', 'action': rsrc.Resource.DELETE, 'root_stack_id': self.stack.id, 'status': rsrc.Resource.COMPLETE}, {'name': 'res2', 'action': rsrc.Resource.DELETE, 'root_stack_id': self.stack.id, 'status': rsrc.Resource.IN_PROGRESS, 'engine_id': 'engine-001'}, {'name': 'res3', 'action': rsrc.Resource.UPDATE, 'root_stack_id': self.stack.id, 'status': rsrc.Resource.IN_PROGRESS, 'engine_id': 'engine-002'}, {'name': 'res4', 'action': rsrc.Resource.CREATE, 'root_stack_id': self.stack.id, 'status': rsrc.Resource.COMPLETE}, {'name': 'res5', 'action': rsrc.Resource.INIT, 'root_stack_id': self.stack.id, 'status': rsrc.Resource.COMPLETE}, {'name': 'res6', 'action': rsrc.Resource.CREATE, 'root_stack_id': self.stack.id, 'status': rsrc.Resource.IN_PROGRESS, 'engine_id': 'engine-001'}, {'name': 'res6'}]
        for val in values:
            create_resource(self.ctx, self.stack, **val)
        engines = db_api.engine_get_all_locked_by_stack(self.ctx, self.stack.id)
        self.assertEqual({'engine-001', 'engine-002'}, engines)