import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def _groups_for_user_data(self):
    number_of_groups = 10
    group_name_data = {5: 'The', 6: 'The Ministry', 9: 'The Ministry of Silly Walks'}
    group_list = self._create_test_data('group', number_of_groups, domain_id=CONF.identity.default_domain_id, name_dict=group_name_data)
    user_list = self._create_test_data('user', 2)
    for group in range(7):
        PROVIDERS.identity_api.add_user_to_group(user_list[0]['id'], group_list[group]['id'])
    for group in range(7, number_of_groups):
        PROVIDERS.identity_api.add_user_to_group(user_list[1]['id'], group_list[group]['id'])
    return (group_list, user_list)