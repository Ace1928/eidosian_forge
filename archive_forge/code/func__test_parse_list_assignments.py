import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import role_assignments
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
def _test_parse_list_assignments(self, entity=None):
    self.test_role_assignment.parse_list_assignments = self.parse_assgmnts
    dict_obj = mock.MagicMock()
    dict_obj.to_dict.side_effect = [{'scope': {'project': {'id': 'fc0fe982401643368ff2eb11d9ca70f1'}}, 'role': {'id': '3b8b253648f44256a457a5073b78021d'}, entity: {'id': '4147558a763046cfb68fb870d58ef4cf'}}, {'role': {'id': '3b8b253648f44258021d6a457a5073b7'}, entity: {'id': '4147558a763046cfb68fb870d58ef4cf'}}]
    self.keystoneclient.role_assignments.list.return_value = [dict_obj, dict_obj]
    kwargs = {'%s_id' % entity: '4147558a763046cfb68fb870d58ef4cf'}
    list_assignments = self.test_role_assignment.parse_list_assignments(**kwargs)
    expected = [{'role': '3b8b253648f44256a457a5073b78021d', 'project': 'fc0fe982401643368ff2eb11d9ca70f1', 'domain': None}, {'role': '3b8b253648f44258021d6a457a5073b7', 'project': None, 'domain': None}]
    self.assertEqual(expected, list_assignments)