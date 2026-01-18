import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class ShowFirewallRule(neutronv20.ShowCommand):
    """Show information of a given firewall rule."""
    resource = 'firewall_rule'