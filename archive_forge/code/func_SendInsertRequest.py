from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import forwarding_rules_utils as utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import exceptions
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
def SendInsertRequest(self, client, forwarding_rule_ref, forwarding_rule):
    """Send forwarding rule insert request."""
    if forwarding_rule_ref.Collection() == 'compute.forwardingRules':
        return client.apitools_client.forwardingRules.Insert(client.messages.ComputeForwardingRulesInsertRequest(forwardingRule=forwarding_rule, project=forwarding_rule_ref.project, region=forwarding_rule_ref.region))
    return client.apitools_client.globalForwardingRules.Insert(client.messages.ComputeGlobalForwardingRulesInsertRequest(forwardingRule=forwarding_rule, project=forwarding_rule_ref.project))