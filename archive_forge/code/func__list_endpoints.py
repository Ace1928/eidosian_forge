import os.path
from oslo_log import log
from keystone.catalog.backends import base
from keystone.common import utils
import keystone.conf
from keystone import exception
def _list_endpoints(self):
    for region_id, region_ref in self.templates.items():
        for service_type, service_ref in region_ref.items():
            for key in service_ref:
                if key.endswith('URL'):
                    interface = key[:-3]
                    endpoint_id = '%s-%s-%s' % (region_id, service_type, interface)
                    yield {'id': endpoint_id, 'service_id': service_type, 'interface': interface, 'url': service_ref[key], 'legacy_endpoint_id': None, 'region_id': region_id, 'enabled': True}