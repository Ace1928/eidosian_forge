from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from dns import rdatatype
from googlecloudsdk.api_lib.dns import import_util
from googlecloudsdk.api_lib.dns import record_types
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def GetLoadBalancerConfigFromUrl(compute_client, compute_messages, forwarding_rule):
    """Attempts to fetch the configuration for the given forwarding rule.

  If forwarding_rule is not the self_link for a forwarding rule,
  one of resources.RequiredFieldOmittedException or
  resources.RequiredFieldOmittedException will be thrown, which must be handled
  by the caller.

  Args:
    compute_client: The configured GCE client for this invocation
    compute_messages: The configured GCE API protobufs for this invocation
    forwarding_rule: The (presumed) selfLink for a GCE forwarding rule

  Returns:
    ForwardingRule, the forwarding rule configuration specified by
    forwarding_rule
  """
    try:
        resource = resources.REGISTRY.Parse(forwarding_rule, collection='compute.forwardingRules').AsDict()
        return compute_client.forwardingRules.Get(compute_messages.ComputeForwardingRulesGetRequest(project=resource['project'], region=resource['region'], forwardingRule=resource['forwardingRule']))
    except (resources.WrongResourceCollectionException, resources.RequiredFieldOmittedException):
        resource = resources.REGISTRY.Parse(forwarding_rule, collection='compute.globalForwardingRules').AsDict()
        return compute_client.globalForwardingRules.Get(compute_messages.ComputeGlobalForwardingRulesGetRequest(project=resource['project'], forwardingRule=resource['forwardingRule']))