import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
class OpenNebula_3_2_NodeDriver(OpenNebula_3_0_NodeDriver):
    """
    OpenNebula.org node driver for OpenNebula.org v3.2.
    """
    name = 'OpenNebula (v3.2)'

    def reboot_node(self, node):
        return self.ex_node_action(node, ACTION.REBOOT)

    def list_sizes(self, location=None):
        """
        Return list of sizes on a provider.

        @inherits: :class:`NodeDriver.list_sizes`

        :return: List of compute node sizes supported by the cloud provider.
        :rtype:  ``list`` of :class:`OpenNebulaNodeSize`
        """
        return self._to_sizes(self.connection.request('/instance_type').object)

    def _to_sizes(self, object):
        """
        Request a list of instance types and convert that list to a list of
        OpenNebulaNodeSize objects.

        Request a list of instance types from the OpenNebula web interface,
        and issue a request to convert each XML object representation of an
        instance type to an OpenNebulaNodeSize object.

        :return: List of instance types.
        :rtype:  ``list`` of :class:`OpenNebulaNodeSize`
        """
        sizes = []
        size_id = 1
        attributes = [('name', str, None), ('ram', int, 'MEMORY'), ('cpu', float, None), ('vcpu', float, None), ('disk', str, None), ('bandwidth', float, None), ('price', float, None)]
        for element in object.findall('INSTANCE_TYPE'):
            size_kwargs = {'id': size_id, 'driver': self}
            values = self._get_attributes_values(attributes=attributes, element=element)
            size_kwargs.update(values)
            size = OpenNebulaNodeSize(**size_kwargs)
            sizes.append(size)
            size_id += 1
        return sizes

    def _get_attributes_values(self, attributes, element):
        values = {}
        for attribute_name, attribute_type, alias in attributes:
            key = alias if alias else attribute_name.upper()
            value = element.findtext(key)
            if value is not None:
                value = attribute_type(value)
            values[attribute_name] = value
        return values