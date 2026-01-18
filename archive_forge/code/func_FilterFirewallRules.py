from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.compute import base_classes as compute_base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions as command_exceptions
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def FilterFirewallRules(firewall_rules):
    """Filters firewall rules that allow ingress to port 22."""
    filtered_firewall_rules = []
    for firewall_rule in firewall_rules:
        if firewall_rule.get('direction') == 'INGRESS':
            allowed_dict = firewall_rule.get('allowed')
            if not allowed_dict:
                continue
            allowed_ports = allowed_dict[0].get('ports')
            if not allowed_ports:
                continue
            if _ContainsPort22(allowed_ports):
                filtered_firewall_rules.append(firewall_rule)
    return filtered_firewall_rules