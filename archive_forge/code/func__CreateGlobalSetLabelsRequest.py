from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import exceptions as fw_exceptions
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
from googlecloudsdk.command_lib.util.args import labels_util
def _CreateGlobalSetLabelsRequest(self, messages, forwarding_rule_ref, forwarding_rule, replacement):
    return messages.ComputeGlobalForwardingRulesSetLabelsRequest(project=forwarding_rule_ref.project, resource=forwarding_rule_ref.Name(), globalSetLabelsRequest=messages.GlobalSetLabelsRequest(labelFingerprint=forwarding_rule.labelFingerprint, labels=replacement))