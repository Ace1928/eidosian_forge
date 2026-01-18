import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def _get_mock_role_query_urls(self, role_data, domain_data=None, project_data=None, group_data=None, user_data=None, use_role_name=False, use_domain_name=False, use_project_name=False, use_group_name=False, use_user_name=False, use_domain_in_query=True):
    """Build uri mocks for querying role assignments"""
    uri_mocks = []
    if domain_data:
        uri_mocks.extend(self.__get('domain', domain_data, 'domain_id' if not use_domain_name else 'domain_name', [], use_name=use_domain_name))
    qs_elements = []
    if domain_data and use_domain_in_query:
        qs_elements = ['domain_id=' + domain_data.domain_id]
    uri_mocks.extend(self.__get('role', role_data, 'role_id' if not use_role_name else 'role_name', [], use_name=use_role_name))
    if user_data:
        uri_mocks.extend(self.__user_mocks(user_data, use_user_name, is_found=True))
    if group_data:
        uri_mocks.extend(self.__get('group', group_data, 'group_id' if not use_group_name else 'group_name', qs_elements, use_name=use_group_name))
    if project_data:
        uri_mocks.extend(self.__get('project', project_data, 'project_id' if not use_project_name else 'project_name', qs_elements, use_name=use_project_name))
    return uri_mocks