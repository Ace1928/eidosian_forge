import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class ShowFirewallPolicy(neutronv20.ShowCommand):
    """Show information of a given firewall policy."""
    resource = 'firewall_policy'