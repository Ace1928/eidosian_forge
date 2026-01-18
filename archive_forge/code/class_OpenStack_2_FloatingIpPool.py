import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
class OpenStack_2_FloatingIpPool:
    """
    Floating IP Pool info.
    """

    def __init__(self, id, name, connection):
        self.id = id
        self.name = name
        self.connection = connection

    def _to_floating_ips(self, obj):
        ip_elements = obj['floatingips']
        return [self._to_floating_ip(ip) for ip in ip_elements]

    def _to_floating_ip(self, obj):
        extra = {}
        extra['port_details'] = obj.get('port_details')
        extra['port_id'] = obj.get('port_id')
        return OpenStack_2_FloatingIpAddress(id=obj['id'], ip_address=obj['floating_ip_address'], pool=self, node_id=None, driver=self.connection.driver, extra=extra)

    def list_floating_ips(self):
        """
        List floating IPs in the pool

        :rtype: ``list`` of :class:`OpenStack_2_FloatingIpAddress`
        """
        url = '/v2.0/floatingips?floating_network_id=%s' % self.id
        return self._to_floating_ips(self.connection.request(url).object)

    def get_floating_ip(self, ip):
        """
        Get specified floating IP from the pool

        :param      ip: floating IP to get
        :type       ip: ``str``

        :rtype: :class:`OpenStack_2_FloatingIpAddress`
        """
        url = '/v2.0/floatingips?floating_network_id=%s' % self.id
        url += '&floating_ip_address=%s' % ip
        floating_ips = self._to_floating_ips(self.connection.request(url).object)
        return floating_ips[0] if floating_ips else None

    def create_floating_ip(self):
        """
        Create new floating IP in the pool

        :rtype: :class:`OpenStack_2_FloatingIpAddress`
        """
        resp = self.connection.request('/v2.0/floatingips', method='POST', data={'floatingip': {'floating_network_id': self.id}})
        data = resp.object['floatingip']
        return OpenStack_2_FloatingIpAddress(id=data['id'], ip_address=data['floating_ip_address'], pool=self, node_id=None, driver=self.connection.driver)

    def delete_floating_ip(self, ip):
        """
        Delete specified floating IP from the pool

        :param      ip: floating IP to remove
        :type       ip: :class:`OpenStack_1_1_FloatingIpAddress`

        :rtype: ``bool``
        """
        resp = self.connection.request('/v2.0/floatingips/%s' % ip.id, method='DELETE')
        return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)

    def __repr__(self):
        return '<OpenStack_2_FloatingIpPool: name=%s>' % self.name