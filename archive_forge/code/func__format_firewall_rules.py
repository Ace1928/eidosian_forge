import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
def _format_firewall_rules(firewall_policy):
    try:
        output = '[' + ',\n '.join([rule for rule in firewall_policy['firewall_rules']]) + ']'
        return output
    except (TypeError, KeyError):
        return ''