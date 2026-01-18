from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import six
def ResolveSubnetURI(project, region, subnet, resource_parser):
    """Resolves the URI of a subnet."""
    if project and region and subnet and resource_parser:
        return six.text_type(resource_parser.Parse(subnet, collection='compute.subnetworks', params={'project': project, 'region': region}))
    return None