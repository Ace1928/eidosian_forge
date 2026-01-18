from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.updater import update_manager
def TransformLastUpdate(r):
    try:
        snapshot = snapshots.ComponentSnapshot.FromURLs(r, command_path='components.repositories.list')
        return snapshot.sdk_definition.LastUpdatedString()
    except (AttributeError, TypeError, snapshots.URLFetchError):
        return 'Unknown'