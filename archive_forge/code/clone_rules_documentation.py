from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.firewall_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.firewall_policies import firewall_policies_utils
from googlecloudsdk.command_lib.compute.firewall_policies import flags
import six
Replace the rules of a Compute Engine organization firewall policy with rules from another policy.

  *{command}* is used to replace the rules of organization firewall policies. An
   organization firewall policy is a set of rules that controls access to
   various resources.
  