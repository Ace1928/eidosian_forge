import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def _create_one_entity(self, entity_type, domain_id, name):
    new_entity = {'name': name, 'domain_id': domain_id}
    if entity_type in ['user', 'group']:
        new_entity = self._create_entity(entity_type)(new_entity)
    else:
        new_entity['id'] = uuid.uuid4().hex
        self._create_entity(entity_type)(new_entity['id'], new_entity)
    return new_entity