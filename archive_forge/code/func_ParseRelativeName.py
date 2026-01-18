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
def ParseRelativeName(self, relative_name, collection, url_unescape=False, api_version=None):
    """Parser relative names. See Resource.RelativeName() method."""
    parser = self.GetParserForCollection(collection, api_version=api_version)
    base_url = GetApiBaseUrl(parser.collection_info.api_name, parser.collection_info.api_version)
    subcollection = parser.collection_info.GetSubcollection(collection)
    return parser.ParseRelativeName(relative_name, base_url, subcollection, url_unescape)