from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def GetResourceRecordSetsRoutingPolicyTypeArg(required=False, deprecated_name=False):
    """Returns --routing-policy-type command line arg value."""
    flag_name = '--routing_policy_type' if deprecated_name else '--routing-policy-type'
    return base.Argument(flag_name, metavar='ROUTING_POLICY_TYPE', required=required, choices=['GEO', 'WRR', 'FAILOVER'], help='Indicates what type of routing policy is being specified. As of this time, this field can take on either "WRR" for weighted round robin, "GEO" for geo location, or "FAILOVER" for a primary-backup configuration. This field cannot be modified - once a policy has a chosen type, the only way to change it is to delete the policy and add a new one with the different type.')