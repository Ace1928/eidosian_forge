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
class DBAPIResourceReplacementTest(common.HeatTestCase):

    def setUp(self):
        super(DBAPIResourceReplacementTest, self).setUp()
        self.useFixture(utils.ForeignKeyConstraintFixture())
        self.ctx = utils.dummy_context()
        self.template = create_raw_template(self.ctx)
        self.user_creds = create_user_creds(self.ctx)
        self.stack = create_stack(self.ctx, self.template, self.user_creds)

    def test_resource_create_replacement(self):
        orig = create_resource(self.ctx, self.stack)
        tmpl_id = create_raw_template(self.ctx).id
        repl = db_api.resource_create_replacement(self.ctx, orig.id, {'name': orig.name, 'replaces': orig.id, 'stack_id': orig.stack_id, 'current_template_id': tmpl_id}, 1, None)
        self.assertIsNotNone(repl)
        self.assertEqual(orig.name, repl.name)
        self.assertNotEqual(orig.id, repl.id)
        self.assertEqual(orig.id, repl.replaces)

    def test_resource_create_replacement_template_gone(self):
        orig = create_resource(self.ctx, self.stack)
        other_ctx = utils.dummy_context()
        tmpl_id = create_raw_template(self.ctx).id
        db_api.raw_template_delete(other_ctx, tmpl_id)
        repl = db_api.resource_create_replacement(self.ctx, orig.id, {'name': orig.name, 'replaces': orig.id, 'stack_id': orig.stack_id, 'current_template_id': tmpl_id}, 1, None)
        self.assertIsNone(repl)

    def test_resource_create_replacement_updated(self):
        orig = create_resource(self.ctx, self.stack)
        other_ctx = utils.dummy_context()
        tmpl_id = create_raw_template(self.ctx).id
        db_api.resource_update_and_save(other_ctx, orig.id, {'atomic_key': 2})
        self.assertRaises(exception.UpdateInProgress, db_api.resource_create_replacement, self.ctx, orig.id, {'name': orig.name, 'replaces': orig.id, 'stack_id': orig.stack_id, 'current_template_id': tmpl_id}, 1, None)

    def test_resource_create_replacement_updated_concurrent(self):
        orig = create_resource(self.ctx, self.stack)
        other_ctx = utils.dummy_context()
        tmpl_id = create_raw_template(self.ctx).id

        def update_atomic_key(*args, **kwargs):
            db_api.resource_update_and_save(other_ctx, orig.id, {'atomic_key': 2})
        self.patchobject(db_api, '_resource_update', new=mock.Mock(wraps=db_api._resource_update, side_effect=update_atomic_key))
        self.assertRaises(exception.UpdateInProgress, db_api.resource_create_replacement, self.ctx, orig.id, {'name': orig.name, 'replaces': orig.id, 'stack_id': orig.stack_id, 'current_template_id': tmpl_id}, 1, None)

    def test_resource_create_replacement_locked(self):
        orig = create_resource(self.ctx, self.stack)
        other_ctx = utils.dummy_context()
        tmpl_id = create_raw_template(self.ctx).id
        db_api.resource_update_and_save(other_ctx, orig.id, {'engine_id': 'a', 'atomic_key': 2})
        self.assertRaises(exception.UpdateInProgress, db_api.resource_create_replacement, self.ctx, orig.id, {'name': orig.name, 'replaces': orig.id, 'stack_id': orig.stack_id, 'current_template_id': tmpl_id}, 1, None)

    def test_resource_create_replacement_locked_concurrent(self):
        orig = create_resource(self.ctx, self.stack)
        other_ctx = utils.dummy_context()
        tmpl_id = create_raw_template(self.ctx).id

        def lock_resource(*args, **kwargs):
            db_api.resource_update_and_save(other_ctx, orig.id, {'engine_id': 'a', 'atomic_key': 2})
        self.patchobject(db_api, '_resource_update', new=mock.Mock(wraps=db_api._resource_update, side_effect=lock_resource))
        self.assertRaises(exception.UpdateInProgress, db_api.resource_create_replacement, self.ctx, orig.id, {'name': orig.name, 'replaces': orig.id, 'stack_id': orig.stack_id, 'current_template_id': tmpl_id}, 1, None)