import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
class DeleteIPsecSiteConnection(neutronv20.DeleteCommand):
    """Delete a given IPsec site connection."""
    resource = 'ipsec_site_connection'
    help_resource = 'IPsec site connection'