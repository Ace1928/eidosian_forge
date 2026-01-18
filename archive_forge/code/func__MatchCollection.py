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
def _MatchCollection(resource_spec, attribute):
    """Gets the collection for an attribute in a resource."""
    resource_collection_info = resource_spec._collection_info
    resource_collection = registry.APICollection(resource_collection_info)
    if resource_collection is None:
        return None
    if attribute == resource_spec.attributes[-1]:
        return resource_collection.name
    attribute_idx = resource_spec.attributes.index(attribute)
    api_name = resource_collection_info.api_name
    resource_collections = registry.GetAPICollections(api_name, resource_collection_info.api_version)
    params = resource_collection.detailed_params[:attribute_idx + 1]
    for c in resource_collections:
        if c.detailed_params == params:
            return c.name