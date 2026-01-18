from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from apitools.base.py import  exceptions as apitools_exc
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis import apis_map
import six
def GetAPICollection(full_collection_name, api_version=None):
    """Gets the given collection for the given API version.

  Args:
    full_collection_name: str, The collection to get including the api name.
    api_version: str, The version string of the API or None to use the default
      for this API.

  Returns:
    APICollection, The requested API collection.

  Raises:
    UnknownCollectionError: If the collection does not exist for the given API
    and version.
  """
    api_name, collection = _SplitFullCollectionName(full_collection_name)
    api_version = _ValidateAndGetDefaultVersion(api_name, api_version)
    collections = GetAPICollections(api_name, api_version)
    for c in collections:
        if c.name == collection:
            return c
    raise UnknownCollectionError(api_name, api_version, collection)