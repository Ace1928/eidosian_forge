from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import user
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _get_mock_user(self):
    value = mock.MagicMock()
    user_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    value.id = user_id
    value.name = 'test_user_1'
    value.default_project_id = 'project_1'
    value.domain_id = 'default'
    value.enabled = True
    value.password_expires_at = '2016-12-10T17:28:49.000000'
    return value