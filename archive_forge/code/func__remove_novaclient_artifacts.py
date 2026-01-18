from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
def _remove_novaclient_artifacts(self, item):
    item.pop('links', None)
    item.pop('NAME_ATTR', None)
    item.pop('HUMAN_ID', None)
    item.pop('human_id', None)
    item.pop('request_ids', None)
    item.pop('x_openstack_request_ids', None)