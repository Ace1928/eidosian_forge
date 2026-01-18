import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
class FormatSoftwareConfigDeploymentTest(common.HeatTestCase):

    def _dummy_software_config(self):
        config = mock.Mock()
        self.now = timeutils.utcnow()
        config.name = 'config_mysql'
        config.group = 'Heat::Shell'
        config.id = str(uuid.uuid4())
        config.created_at = self.now
        config.config = {'inputs': [{'name': 'bar'}], 'outputs': [{'name': 'result'}], 'options': {}, 'config': '#!/bin/bash\n'}
        config.tenant = str(uuid.uuid4())
        return config

    def _dummy_software_deployment(self):
        config = self._dummy_software_config()
        deployment = mock.Mock()
        deployment.config = config
        deployment.id = str(uuid.uuid4())
        deployment.server_id = str(uuid.uuid4())
        deployment.input_values = {'bar': 'baaaaa'}
        deployment.output_values = {'result': '0'}
        deployment.action = 'INIT'
        deployment.status = 'COMPLETE'
        deployment.status_reason = 'Because'
        deployment.created_at = config.created_at
        deployment.updated_at = config.created_at
        return deployment

    def test_format_software_config(self):
        config = self._dummy_software_config()
        result = api.format_software_config(config)
        self.assertIsNotNone(result)
        self.assertEqual([{'name': 'bar'}], result['inputs'])
        self.assertEqual([{'name': 'result'}], result['outputs'])
        self.assertEqual([{'name': 'result'}], result['outputs'])
        self.assertEqual({}, result['options'])
        self.assertEqual(heat_timeutils.isotime(self.now), result['creation_time'])
        self.assertNotIn('project', result)
        result = api.format_software_config(config, include_project=True)
        self.assertIsNotNone(result)
        self.assertEqual([{'name': 'bar'}], result['inputs'])
        self.assertEqual([{'name': 'result'}], result['outputs'])
        self.assertEqual([{'name': 'result'}], result['outputs'])
        self.assertEqual({}, result['options'])
        self.assertEqual(heat_timeutils.isotime(self.now), result['creation_time'])
        self.assertIn('project', result)

    def test_format_software_config_none(self):
        self.assertIsNone(api.format_software_config(None))

    def test_format_software_deployment(self):
        deployment = self._dummy_software_deployment()
        result = api.format_software_deployment(deployment)
        self.assertIsNotNone(result)
        self.assertEqual(deployment.id, result['id'])
        self.assertEqual(deployment.config.id, result['config_id'])
        self.assertEqual(deployment.server_id, result['server_id'])
        self.assertEqual(deployment.input_values, result['input_values'])
        self.assertEqual(deployment.output_values, result['output_values'])
        self.assertEqual(deployment.action, result['action'])
        self.assertEqual(deployment.status, result['status'])
        self.assertEqual(deployment.status_reason, result['status_reason'])
        self.assertEqual(heat_timeutils.isotime(self.now), result['creation_time'])
        self.assertEqual(heat_timeutils.isotime(self.now), result['updated_time'])

    def test_format_software_deployment_none(self):
        self.assertIsNone(api.format_software_deployment(None))