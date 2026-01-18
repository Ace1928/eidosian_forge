from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as sdk_core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
class TPUQueuedResource(object):
    """Helper to get TPU Queued Resources."""

    def __init__(self, release_track):
        if release_track == base.ReleaseTrack.ALPHA:
            self._api_version = 'v2alpha1'
        else:
            self._api_version = 'v2'
        self.client = apis.GetClientInstance('tpu', self._api_version)
        self.messages = apis.GetMessagesModule('tpu', self._api_version)

    def GetMessages(self):
        return self.messages

    def Get(self, name, zone):
        """Retrieves the Queued Resource in the given project and zone."""
        project = properties.VALUES.core.project.Get(required=True)
        fully_qualified_queued_resource_name_ref = resources.REGISTRY.Parse(name, params={'locationsId': zone, 'projectsId': project}, collection='tpu.projects.locations.queuedResources', api_version=self._api_version)
        request = self.messages.TpuProjectsLocationsQueuedResourcesGetRequest(name=fully_qualified_queued_resource_name_ref.RelativeName())
        return self.client.projects_locations_queuedResources.Get(request)