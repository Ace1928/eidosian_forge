from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.util import resource as resource_lib  # pylint: disable=unused-import
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.concepts import resource_parameter_info
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def _GetCompleterCollectionInfo(resource_spec, attribute):
    """Gets collection info for an attribute in a resource."""
    api_version = None
    collection = _MatchCollection(resource_spec, attribute)
    if collection:
        full_collection_name = resource_spec._collection_info.api_name + '.' + collection
        api_version = resource_spec._collection_info.api_version
    elif attribute.name == 'project':
        full_collection_name = 'cloudresourcemanager.projects'
    else:
        return None
    return resources.REGISTRY.GetCollectionInfo(full_collection_name, api_version=api_version)