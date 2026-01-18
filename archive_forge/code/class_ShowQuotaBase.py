import abc
import argparse
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ShowQuotaBase(neutronV20.NeutronCommand, show.ShowOne):
    """Base class to show quotas of a given tenant."""
    resource = 'quota'

    @abc.abstractmethod
    def retrieve_data(self, tenant_id, neutron_client):
        """Retrieve data using neutron client for the given tenant."""

    def get_parser(self, prog_name):
        parser = super(ShowQuotaBase, self).get_parser(prog_name)
        parser.add_argument('--tenant-id', metavar='tenant-id', help=_('The owner tenant ID.'))
        parser.add_argument('--tenant_id', help=argparse.SUPPRESS)
        parser.add_argument('pos_tenant_id', help=argparse.SUPPRESS, nargs='?')
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        tenant_id = get_tenant_id(parsed_args, neutron_client)
        data = self.retrieve_data(tenant_id, neutron_client)
        if self.resource in data:
            return zip(*sorted(data[self.resource].items()))
        return