from unittest import mock
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.keystone import project
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _get_mock_project(self):
    value = mock.MagicMock()
    project_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    value.id = project_id
    value.name = 'test_project_1'
    value.domain_id = 'default'
    value.enabled = True
    value.parent_id = 'my_father'
    value.is_domain = False
    return value