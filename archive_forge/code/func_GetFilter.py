from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.core import properties
import six
def GetFilter(image_ref, holder):
    """Get the filter of occurrences request for container analysis API."""
    filters = ['kind = "PACKAGE_MANAGER"', 'has_prefix(resource_url,"https://compute.googleapis.com/compute/")']
    client = holder.client
    resource_parser = holder.resources
    if image_ref:
        image_expander = image_utils.ImageExpander(client, resource_parser)
        self_link, image = image_expander.ExpandImageFlag(user_project=properties.VALUES.core.project.Get(), image=image_ref.image, image_project=image_ref.project, return_image_resource=True)
        image_url = self_link + '/id/' + six.text_type(image.id)
        filters.append('has_prefix(resource_url,"{}")'.format(image_url))
    return ' AND '.join(filters)