from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def _HasProject(self, collection):
    collection_info = self._resources.GetCollectionInfo(collection, self._version)
    return 'projects' in collection_info.path or 'projects' in collection_info.base_url