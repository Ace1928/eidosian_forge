from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.filestore.locations import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def UriFunc(resource):
    registry = filestore_client.GetFilestoreRegistry(filestore_client.ALPHA_API_VERSION)
    ref = registry.Parse(resource.name, collection=filestore_client.LOCATIONS_COLLECTION)
    return ref.SelfLink()