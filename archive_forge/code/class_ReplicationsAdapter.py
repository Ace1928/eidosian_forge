from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
class ReplicationsAdapter(object):
    """Adapter for the Cloud NetApp Files API Replication resource."""

    def __init__(self):
        self.release_track = base.ReleaseTrack.GA
        self.client = netapp_api_util.GetClientInstance(release_track=self.release_track)
        self.messages = netapp_api_util.GetMessagesModule(release_track=self.release_track)

    def ParseDestinationVolumeParameters(self, replication, destination_volume_parameters):
        """Parses Destination Volume Parameters for Replication into a config.

    Args:
      replication: The Cloud Netapp Volumes Replication object.
      destination_volume_parameters: The Destination Volume Parameters message
        object.

    Returns:
      Replication message populated with Destination Volume Parameters values.
    """
        if not destination_volume_parameters:
            return
        parameters = self.messages.DestinationVolumeParameters()
        for key, val in destination_volume_parameters.items():
            if key == 'storage_pool':
                parameters.storagePool = val
            elif key == 'volume_id':
                parameters.volumeId = val
            elif key == 'share_name':
                parameters.shareName = val
            elif key == 'description':
                parameters.description = val
            else:
                log.warning('The attribute {} is not recognized.'.format(key))
        replication.destinationVolumeParameters = parameters

    def UpdateReplication(self, replication_ref, replication_config, update_mask):
        """Send a Patch request for the Cloud NetApp Volume Replication."""
        update_request = self.messages.NetappProjectsLocationsVolumesReplicationsPatchRequest(replication=replication_config, name=replication_ref.RelativeName(), updateMask=update_mask)
        update_op = self.client.projects_locations_volumes_replications.Patch(update_request)
        return update_op

    def ParseUpdatedReplicationConfig(self, replication_config, description=None, labels=None, replication_schedule=None):
        """Parse update information into an updated Replication message."""
        if description is not None:
            replication_config.description = description
        if labels is not None:
            replication_config.labels = labels
        if replication_schedule is not None:
            replication_config.replicationSchedule = replication_schedule
        return replication_config

    def ResumeReplication(self, replication_ref):
        """Send a resume request for the Cloud NetApp Volume Replication."""
        resume_request = self.messages.NetappProjectsLocationsVolumesReplicationsResumeRequest(name=replication_ref.RelativeName())
        return self.client.projects_locations_volumes_replications.Resume(resume_request)

    def ReverseReplication(self, replication_ref):
        """Send a reverse request for the Cloud NetApp Volume Replication."""
        reverse_request = self.messages.NetappProjectsLocationsVolumesReplicationsReverseDirectionRequest(name=replication_ref.RelativeName())
        return self.client.projects_locations_volumes_replications.ReverseDirection(reverse_request)

    def StopReplication(self, replication_ref, force):
        """Send a stop request for the Cloud NetApp Volume Replication."""
        stop_request = self.messages.NetappProjectsLocationsVolumesReplicationsStopRequest(name=replication_ref.RelativeName(), stopReplicationRequest=self.messages.StopReplicationRequest(force=force))
        return self.client.projects_locations_volumes_replications.Stop(stop_request)