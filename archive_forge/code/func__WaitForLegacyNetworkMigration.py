from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.networks import flags
from googlecloudsdk.command_lib.compute.networks import network_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
def _WaitForLegacyNetworkMigration(self, operation_poller, operation_ref):
    progress_stages = []
    for key, label in self.MIGRATION_STAGES.items():
        progress_stages.append(progress_tracker.Stage(label, key=key))
    tracker = progress_tracker.StagedProgressTracker(message='Migrating Network from Legacy to Custom Mode', stages=progress_stages)
    first_status_message = list(self.MIGRATION_STAGES.keys())[0]
    tracker.last_status_message = first_status_message
    return waiter.WaitFor(poller=operation_poller, operation_ref=operation_ref, custom_tracker=tracker, tracker_update_func=self._LegacyNetworkMigrationTrackerUpdateFunc)