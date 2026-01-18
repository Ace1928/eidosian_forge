from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags as security_policy_flags
from googlecloudsdk.command_lib.compute.security_policies.rules import flags
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class RemovePreconfigWafExclusionAlpha(RemovePreconfigWafExclusionBeta):
    """Remove an exclusion configuration for preconfigured WAF evaluation from a security policy rule.

  *{command}* is used to remove an exclusion configuration for preconfigured WAF
  evaluation from a security policy rule.

  Note that request field exclusions are associated with a target, which can be
  a single rule set, or a rule set plus a list of rule IDs under the rule set.

  It is possible to remove request field exclusions at 3 levels:
  - Remove specific request field exclusions that are associated with a matching
    target.
  - Remove all the request field exclusions that are associated with a matching
    target.
  - Remove all the request field exclusions that are configured under the
    security policy rule, regardless of the target.

  ## EXAMPLES

  To remove specific request field exclusions that are associated with the
  target of 'sqli-stable': ['owasp-crs-v030001-id942110-sqli',
  'owasp-crs-v030001-id942120-sqli'], run:

    $ {command} 1000 \\
       --security-policy=my-policy \\
       --target-rule-set=sqli-stable \\
       --target-rule-ids=owasp-crs-v030001-id942110-sqli,owasp-crs-v030001-id942120-sqli
       \\
       --request-header-to-exclude=op=EQUALS,val=abc \\
       --request-header-to-exclude=op=STARTS_WITH,val=xyz \\
       --request-uri-to-exclude=op=EQUALS_ANY

  To remove all the request field exclusions that are associated with the target
  of 'sqli-stable': ['owasp-crs-v030001-id942110-sqli',
  'owasp-crs-v030001-id942120-sqli'], run:

    $ {command} 1000 \\
       --security-policy=my-policy \\
       --target-rule-set=sqli-stable \\
       --target-rule-ids=owasp-crs-v030001-id942110-sqli,owasp-crs-v030001-id942120-sqli

  To remove all the request field exclusions that are associated with the target
  of 'sqli-stable': [], run:

    $ {command} 1000 \\
       --security-policy=my-policy \\
       --target-rule-set=sqli-stable

  To remove all the request field exclusions that are configured under the
  security policy rule, regardless of the target, run:

    $ {command} 1000 \\
       --security-policy=my-policy \\
       --target-rule-set=*
  """