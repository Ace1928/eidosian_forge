import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _stub_nova_server_get(self, not_found=False):
    mock_server = mock.MagicMock()
    mock_server.image = {'id': 'dd619705-468a-4f7d-8a06-b84794b3561a'}
    mock_server.flavor = {'id': '1'}
    mock_server.key_name = 'test'
    mock_server.security_groups = [{u'name': u'hth_test'}]
    if not_found:
        self.patchobject(nova.NovaClientPlugin, 'get_server', side_effect=exception.EntityNotFound(entity='Server', name='5678'))
    else:
        self.patchobject(nova.NovaClientPlugin, 'get_server', return_value=mock_server)