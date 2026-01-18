from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import six
def ResolveForwardingRuleURI(project, region, forwarding_rule, resource_parser):
    """Resolves the URI of a forwarding rule."""
    if project and region and forwarding_rule and resource_parser:
        return six.text_type(resource_parser.Parse(forwarding_rule, collection='compute.forwardingRules', params={'project': project, 'region': region}))
    return None