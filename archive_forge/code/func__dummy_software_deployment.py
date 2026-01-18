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