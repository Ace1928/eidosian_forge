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
def GetCollectionInfo(self, collection_name, api_version=None):
    api_name = _APINameFromCollection(collection_name)
    api_version = self.RegisterApiByName(api_name, api_version=api_version)
    parser = self.parsers_by_collection.get(api_name, {}).get(api_version, {}).get(collection_name, None)
    if parser is None:
        raise InvalidCollectionException(collection_name, api_version)
    return parser.collection_info