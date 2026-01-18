from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import allocation as _allocation
from openstack.baremetal.v1 import chassis as _chassis
from openstack.baremetal.v1 import conductor as _conductor
from openstack.baremetal.v1 import deploy_templates as _deploytemplates
from openstack.baremetal.v1 import driver as _driver
from openstack.baremetal.v1 import node as _node
from openstack.baremetal.v1 import port as _port
from openstack.baremetal.v1 import port_group as _portgroup
from openstack.baremetal.v1 import volume_connector as _volumeconnector
from openstack.baremetal.v1 import volume_target as _volumetarget
from openstack import exceptions
from openstack import proxy
from openstack import utils
def _get_with_fields(self, resource_type, value, fields=None):
    """Fetch a bare metal resource.

        :param resource_type: The type of resource to get.
        :type resource_type: :class:`~openstack.resource.Resource`
        :param value: The value to get. Can be either the ID of a
            resource or a :class:`~openstack.resource.Resource`
            subclass.
        :param fields: Limit the resource fields to fetch.

        :returns: The result of the ``fetch``
        :rtype: :class:`~openstack.resource.Resource`
        """
    res = self._get_resource(resource_type, value)
    kwargs = {}
    if fields:
        kwargs['fields'] = _common.fields_type(fields, resource_type)
    return res.fetch(self, error_message='No {resource_type} found for {value}'.format(resource_type=resource_type.__name__, value=value), **kwargs)