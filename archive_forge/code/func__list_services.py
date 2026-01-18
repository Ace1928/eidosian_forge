import os.path
from oslo_log import log
from keystone.catalog.backends import base
from keystone.common import utils
import keystone.conf
from keystone import exception
def _list_services(self, hints):
    for region_ref in self.templates.values():
        for service_type, service_ref in region_ref.items():
            yield {'id': service_type, 'enabled': True, 'name': service_ref.get('name', ''), 'description': service_ref.get('description', ''), 'type': service_type}