import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
class TestClusterTemplate(magnum_fakes.TestMagnumClientOSCV1):
    default_create_args = {'coe': 'kubernetes', 'dns_nameserver': '8.8.8.8', 'docker_storage_driver': 'overlay2', 'docker_volume_size': None, 'external_network_id': 'public', 'fixed_network': None, 'fixed_subnet': None, 'flavor_id': 'm1.medium', 'http_proxy': None, 'https_proxy': None, 'image_id': 'fedora-atomic-latest', 'keypair_id': None, 'labels': {}, 'master_flavor_id': None, 'master_lb_enabled': False, 'name': 'fake-ct-1', 'network_driver': None, 'no_proxy': None, 'public': False, 'registry_enabled': False, 'server_type': 'vm', 'tls_disabled': False, 'volume_driver': None}

    def setUp(self):
        super(TestClusterTemplate, self).setUp()
        self.cluster_templates_mock = self.app.client_manager.container_infra.cluster_templates