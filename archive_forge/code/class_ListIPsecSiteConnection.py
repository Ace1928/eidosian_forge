import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
class ListIPsecSiteConnection(neutronv20.ListCommand):
    """List IPsec site connections that belong to a given tenant."""
    resource = 'ipsec_site_connection'
    _formatters = {'peer_cidrs': _format_peer_cidrs}
    list_columns = ['id', 'name', 'peer_address', 'auth_mode', 'status']
    pagination_support = True
    sorting_support = True