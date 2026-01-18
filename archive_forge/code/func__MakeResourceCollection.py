from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import OrderedDict
import json
import re
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core.util import files
import six
def _MakeResourceCollection(self, api_version, collection_name, path, flat_path=None):
    """Make resource collection object given its name and path."""
    if flat_path == path:
        flat_path = None
    url = self.base_url + path
    url_api_name, url_api_version, path = resource_util.SplitEndpointUrl(url)
    if url_api_version != api_version:
        raise UnsupportedDiscoveryDoc('Collection {0} for version {1}/{2} is using url {3} with version {4}'.format(collection_name, self.api_name, api_version, url, url_api_version))
    if flat_path:
        _, _, flat_path = resource_util.SplitEndpointUrl(self.base_url + flat_path)
    url = url[:-len(path)]
    return resource_util.CollectionInfo(url_api_name, api_version, url, self.docs_url, collection_name, path, {DEFAULT_PATH_NAME: flat_path} if flat_path else {}, resource_util.GetParamsFromPath(path))