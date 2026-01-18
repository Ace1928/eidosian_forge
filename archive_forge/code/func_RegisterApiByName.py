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
def RegisterApiByName(self, api_name, api_version=None):
    """Register the given API if it has not been registered already.

    Args:
      api_name: str, The API name.
      api_version: str, The API version, None for the default version.
    Returns:
      api version which was registered.
    """
    registered_version = self.registered_apis.get(api_name, None)
    if api_version is None:
        if registered_version:
            api_version = registered_version
        else:
            api_version = apis_internal._GetDefaultVersion(api_name)
    if api_version not in self.parsers_by_collection.get(api_name, {}):
        for collection in apis_internal._GetApiCollections(api_name, api_version):
            self._RegisterCollection(collection)
    self.registered_apis[api_name] = api_version
    return api_version