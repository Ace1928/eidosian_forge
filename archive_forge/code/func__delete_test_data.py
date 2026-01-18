import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def _delete_test_data(self, entity_type, entity_list):
    for entity in entity_list:
        try:
            self._delete_entity(entity_type)(entity['id'])
        except exception.Forbidden:
            break