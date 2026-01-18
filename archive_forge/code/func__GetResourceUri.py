from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import connection_context
@classmethod
def _GetResourceUri(cls, resource):
    """Get uri for resource.

    This is a @classmethod because this method is called by
    googlecloudsdk.calliope.display_info.DisplayInfo outside of a List instance.

    Args:
      resource: a googlecloudsdk.command_lib.run.k8s_object.KubernetesObject
        object

    Returns:
      uri: str of the resource's uri
    """
    complete_endpoint = cls.complete_api_endpoint
    if not complete_endpoint:
        try:
            region = resource.locationId
        except AttributeError:
            region = resource.region
        complete_endpoint = connection_context.DeriveRegionalEndpoint(cls.partial_api_endpoint, region)
    return '{}/{}'.format(complete_endpoint.rstrip('/'), getattr(resource, 'self_link', ''))