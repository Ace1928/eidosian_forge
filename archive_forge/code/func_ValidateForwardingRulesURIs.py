from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def ValidateForwardingRulesURIs(unused_ref, args, request):
    """Checks if all provided forwarding rules URIs are in correct format."""
    flags = ['destination_forwarding_rule']
    forwarding_rule_pattern = re.compile('projects/(?:[a-z][a-z0-9-\\.:]*[a-z0-9])/(global|regions/[-\\w]+)/forwardingRules/[-\\w]+')
    for flag in flags:
        if not args.IsSpecified(flag):
            continue
        forwarding_rule = getattr(args, flag)
        if not forwarding_rule_pattern.match(forwarding_rule):
            raise InvalidInputError('Invalid value for flag {flag}: {forwarding_rule}\n' + 'Expected forwarding rule in one of the following formats:\n' + '  projects/my-project/global/forwardingRules/my-forwarding-rule\n' + '  projects/my-project/regions/us-central1/forwardingRules/my-forwarding-rule')
    return request