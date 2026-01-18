import argparse
from unittest import mock
from osc_lib.tests import utils
from oslo_utils import uuidutils
from openstack.network.v2 import sfc_flow_classifier as flow_classifier
from openstack.network.v2 import sfc_port_chain as port_chain
from openstack.network.v2 import sfc_port_pair as port_pair
from openstack.network.v2 import sfc_port_pair_group as port_pair_group
from openstack.network.v2 import sfc_service_graph as service_graph
class FakeSfcPortChain(object):
    """Fake port chain attributes."""

    @staticmethod
    def create_port_chain(attrs=None):
        """Create a fake port chain.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A Dictionary with faking port chain attributes
        """
        attrs = attrs or {}
        port_chain_attrs = {'id': uuidutils.generate_uuid(), 'name': 'port-chain-name', 'description': 'description', 'port_pair_groups': uuidutils.generate_uuid(), 'flow_classifiers': uuidutils.generate_uuid(), 'chain_parameters': {'correlation': 'mpls', 'symmetric': False}, 'project_id': uuidutils.generate_uuid()}
        port_chain_attrs.update(attrs)
        return port_chain.SfcPortChain(**port_chain_attrs)

    @staticmethod
    def create_port_chains(attrs=None, count=1):
        """Create multiple port chains.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of port chains to fake
        :return:
            A list of dictionaries faking the port chains.
        """
        port_chains = []
        for _ in range(count):
            port_chains.append(FakeSfcPortChain.create_port_chain(attrs))
        return port_chains