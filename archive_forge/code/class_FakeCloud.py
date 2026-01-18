from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
class FakeCloud:
    config = FakeConfig()
    name = 'test-name'
    private = False
    force_ipv4 = False
    service_val = True
    _unused = 'useless'
    _local_ipv6 = True

    def get_flavor_name(self, id):
        return 'test-flavor-name'

    def get_image_name(self, id):
        return 'test-image-name'

    def get_volumes(self, server):
        return []

    def has_service(self, service_name):
        return self.service_val

    def use_internal_network(self):
        return True

    def use_external_network(self):
        return True

    def get_internal_networks(self):
        return []

    def get_external_networks(self):
        return []

    def get_internal_ipv4_networks(self):
        return []

    def get_external_ipv4_networks(self):
        return []

    def get_internal_ipv6_networks(self):
        return []

    def get_external_ipv6_networks(self):
        return []

    def list_server_security_groups(self, server):
        return []

    def get_default_network(self):
        return None