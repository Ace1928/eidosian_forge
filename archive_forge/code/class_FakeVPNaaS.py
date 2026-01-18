import collections
from unittest import mock
import uuid
from openstack.network.v2 import vpn_endpoint_group as vpn_epg
from openstack.network.v2 import vpn_ike_policy as vpn_ikep
from openstack.network.v2 import vpn_ipsec_policy as vpn_ipsecp
from openstack.network.v2 import vpn_ipsec_site_connection as vpn_sitec
from openstack.network.v2 import vpn_service
class FakeVPNaaS(object):

    def create(self, attrs={}):
        """Create a fake vpnaas resources

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A OrderedDict faking the vpnaas resource
        """
        self.ordered.update(attrs)
        if 'IKEPolicy' == self.__class__.__name__:
            return vpn_ikep.VpnIkePolicy(**self.ordered)
        if 'IPSecPolicy' == self.__class__.__name__:
            return vpn_ipsecp.VpnIpsecPolicy(**self.ordered)
        if 'VPNService' == self.__class__.__name__:
            return vpn_service.VpnService(**self.ordered)
        if 'EndpointGroup' == self.__class__.__name__:
            return vpn_epg.VpnEndpointGroup(**self.ordered)

    def bulk_create(self, attrs=None, count=2):
        """Create multiple fake vpnaas resources

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of vpnaas resources to fake
        :return:
            A list of dictionaries faking the vpnaas resources
        """
        return [self.create(attrs=attrs) for i in range(0, count)]

    def get(self, attrs=None, count=2):
        """Get multiple fake vpnaas resources

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of vpnaas resources to fake
        :return:
            A list of dictionaries faking the vpnaas resource
        """
        if attrs is None:
            self.attrs = self.bulk_create(count=count)
        return mock.Mock(side_effect=attrs)