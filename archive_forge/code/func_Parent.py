from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
import uritemplate
def Parent(self, parent_collection=None):
    """Gets a reference to the parent of this resource.

    If parent_collection is not given, we attempt to automatically determine it
    by finding the collection within the same API that has the correct set of
    URI parameters for what we expect. If the parent collection cannot be
    automatically determined, it can be specified manually.

    Args:
      parent_collection: str, The full collection name of the parent resource.
        Only required if it cannot be automatically determined.

    Raises:
      ParentCollectionResolutionException: If the parent collection cannot be
        determined or doesn't exist.
      ParentCollectionMismatchException: If the given or auto-resolved parent
       collection does not have the expected URI parameters.

    Returns:
      Resource, The reference to the parent resource.
    """
    parent_params = self._params[:-1]
    all_collections = self._registry.parsers_by_collection[self._collection_info.api_name][self._collection_info.api_version]
    if parent_collection:
        try:
            parent_parser = all_collections[parent_collection]
        except KeyError:
            raise UnknownCollectionException(parent_collection)
        actual_parent_params = parent_parser.collection_info.GetParams('')
        if actual_parent_params != parent_params:
            raise ParentCollectionMismatchException(self.Collection(), parent_collection, parent_params, actual_parent_params)
    else:
        for collection, parser in six.iteritems(all_collections):
            if parser.collection_info.GetParams('') == parent_params:
                parent_collection = collection
                break
        if not parent_collection:
            raise ParentCollectionResolutionException(self.Collection(), parent_params)
    parent_param_values = {k: getattr(self, k) for k in parent_params}
    ref = self._registry.Parse(None, parent_param_values, collection=parent_collection)
    return ref