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
class DBAPIEventTest(common.HeatTestCase):

    def setUp(self):
        super(DBAPIEventTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.template = create_raw_template(self.ctx)
        self.user_creds = create_user_creds(self.ctx)

    def test_event_create(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        event = create_event(self.ctx, stack_id=stack.id)
        with db_api.context_manager.reader.using(self.ctx):
            ret_event = self.ctx.session.query(models.Event).filter_by(id=event.id).options(orm.joinedload(models.Event.rsrc_prop_data)).first()
        self.assertIsNotNone(ret_event)
        self.assertEqual(stack.id, ret_event.stack_id)
        self.assertEqual('create', ret_event.resource_action)
        self.assertEqual('complete', ret_event.resource_status)
        self.assertEqual('res', ret_event.resource_name)
        self.assertEqual(UUID1, ret_event.physical_resource_id)
        self.assertEqual('create_complete', ret_event.resource_status_reason)
        self.assertEqual({'foo2': 'ev_bar'}, ret_event.rsrc_prop_data.data)

    def test_event_get_all_by_tenant(self):
        stack1 = create_stack(self.ctx, self.template, self.user_creds, tenant='tenant1')
        stack2 = create_stack(self.ctx, self.template, self.user_creds, tenant='tenant2')
        values = [{'stack_id': stack1.id, 'resource_name': 'res1'}, {'stack_id': stack1.id, 'resource_name': 'res2'}, {'stack_id': stack2.id, 'resource_name': 'res3'}]
        [create_event(self.ctx, **val) for val in values]
        self.ctx.project_id = 'tenant1'
        events = db_api.event_get_all_by_tenant(self.ctx)
        self.assertEqual(2, len(events))
        marker = events[0].uuid
        expected = events[1].uuid
        events = db_api.event_get_all_by_tenant(self.ctx, marker=marker)
        self.assertEqual(1, len(events))
        self.assertEqual(expected, events[0].uuid)
        events = db_api.event_get_all_by_tenant(self.ctx, limit=1)
        self.assertEqual(1, len(events))
        filters = {'resource_name': 'res2'}
        events = db_api.event_get_all_by_tenant(self.ctx, filters=filters)
        self.assertEqual(1, len(events))
        self.assertEqual('res2', events[0].resource_name)
        sort_keys = 'resource_type'
        events = db_api.event_get_all_by_tenant(self.ctx, sort_keys=sort_keys)
        self.assertEqual(2, len(events))
        self.ctx.project_id = 'tenant2'
        events = db_api.event_get_all_by_tenant(self.ctx)
        self.assertEqual(1, len(events))

    def test_event_get_all_by_stack(self):
        stack1 = create_stack(self.ctx, self.template, self.user_creds)
        stack2 = create_stack(self.ctx, self.template, self.user_creds)
        values = [{'stack_id': stack1.id, 'resource_name': 'res1'}, {'stack_id': stack1.id, 'resource_name': 'res2'}, {'stack_id': stack2.id, 'resource_name': 'res3'}]
        [create_event(self.ctx, **val) for val in values]
        self.ctx.project_id = 'tenant1'
        events = db_api.event_get_all_by_stack(self.ctx, stack1.id)
        self.assertEqual(2, len(events))
        self.ctx.project_id = 'tenant2'
        events = db_api.event_get_all_by_stack(self.ctx, stack2.id)
        self.assertEqual(1, len(events))

    def test_event_count_all_by_stack(self):
        stack1 = create_stack(self.ctx, self.template, self.user_creds)
        stack2 = create_stack(self.ctx, self.template, self.user_creds)
        values = [{'stack_id': stack1.id, 'resource_name': 'res1'}, {'stack_id': stack1.id, 'resource_name': 'res2'}, {'stack_id': stack2.id, 'resource_name': 'res3'}]
        [create_event(self.ctx, **val) for val in values]
        self.assertEqual(2, db_api.event_count_all_by_stack(self.ctx, stack1.id))
        self.assertEqual(1, db_api.event_count_all_by_stack(self.ctx, stack2.id))