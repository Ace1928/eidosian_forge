import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from heat.common import crypt
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import service
from heat.engine import service_software_config
from heat.engine import software_config_io as swc_io
from heat.objects import resource as resource_objects
from heat.objects import software_config as software_config_object
from heat.objects import software_deployment as software_deployment_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def _create_software_deployment(self, config_id=None, input_values=None, action='INIT', status='COMPLETE', status_reason='', config_group=None, server_id=str(uuid.uuid4()), config_name=None, stack_user_project_id=None):
    input_values = input_values or {}
    if config_id is None:
        config = self._create_software_config(group=config_group, name=config_name)
        config_id = config['id']
    return self.engine.create_software_deployment(self.ctx, server_id, config_id, input_values, action, status, status_reason, stack_user_project_id)