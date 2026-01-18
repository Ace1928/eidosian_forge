import argparse
import logging
from osc_lib.cli import parseractions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
def _format_network_security_group_rule(self, rule):
    """Transform the SDK SecurityGroupRule object to a dict

        The SDK object gets in the way of reformatting columns...
        Create port_range column from port_range_min and port_range_max
        """
    rule = rule.to_dict()
    rule['port_range'] = network_utils.format_network_port_range(rule)
    rule['remote_ip_prefix'] = network_utils.format_remote_ip_prefix(rule)
    return rule