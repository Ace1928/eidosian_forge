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
class DBAPIRawTemplateTest(common.HeatTestCase):

    def setUp(self):
        super(DBAPIRawTemplateTest, self).setUp()
        self.ctx = utils.dummy_context()

    def test_raw_template_create(self):
        t = template_format.parse(wp_template)
        tp = create_raw_template(self.ctx, template=t)
        self.assertIsNotNone(tp.id)
        self.assertEqual(t, tp.template)

    def test_raw_template_get(self):
        t = template_format.parse(wp_template)
        tp = create_raw_template(self.ctx, template=t)
        template = db_api.raw_template_get(self.ctx, tp.id)
        self.assertEqual(tp.id, template.id)
        self.assertEqual(tp.template, template.template)

    def test_raw_template_update(self):
        another_wp_template = '\n        {\n          "AWSTemplateFormatVersion" : "2010-09-09",\n          "Description" : "WordPress",\n          "Parameters" : {\n            "KeyName" : {\n              "Description" : "KeyName",\n              "Type" : "String",\n              "Default" : "test"\n            }\n          },\n          "Resources" : {\n            "WebServer": {\n              "Type": "AWS::EC2::Instance",\n              "Properties": {\n                "ImageId" : "fedora-20.x86_64.qcow2",\n                "InstanceType"   : "m1.xlarge",\n                "KeyName"        : "test",\n                "UserData"       : "wordpress"\n              }\n            }\n          }\n        }\n        '
        new_t = template_format.parse(another_wp_template)
        new_files = {'foo': 'bar', 'myfile': 'file:///home/somefile'}
        new_values = {'template': new_t, 'files': new_files}
        orig_tp = create_raw_template(self.ctx)
        updated_tp = db_api.raw_template_update(self.ctx, orig_tp.id, new_values)
        self.assertEqual(orig_tp.id, updated_tp.id)
        self.assertEqual(new_t, updated_tp.template)
        self.assertEqual(new_files, updated_tp.files)

    def test_raw_template_delete(self):
        t = template_format.parse(wp_template)
        tp = create_raw_template(self.ctx, template=t)
        db_api.raw_template_delete(self.ctx, tp.id)
        self.assertRaises(exception.NotFound, db_api.raw_template_get, self.ctx, tp.id)