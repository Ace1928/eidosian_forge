import concurrent.futures
import urllib.parse
import keystoneauth1.exceptions
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1._proxy import Proxy
def _get_object_endpoint(self, container, obj=None, query_string=None):
    endpoint = urllib.parse.quote(container)
    if obj:
        endpoint = '{endpoint}/{object}'.format(endpoint=endpoint, object=urllib.parse.quote(obj))
    if query_string:
        endpoint = '{endpoint}?{query_string}'.format(endpoint=endpoint, query_string=query_string)
    return endpoint