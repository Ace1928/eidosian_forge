import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def _get_volume_kwargs(self, kwargs):
    name = kwargs.pop('name', kwargs.pop('display_name', None))
    description = kwargs.pop('description', kwargs.pop('display_description', None))
    if name:
        kwargs['name'] = name
    if description:
        kwargs['description'] = description
    return kwargs