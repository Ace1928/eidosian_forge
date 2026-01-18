from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
import six
def SetFromSnapshot(self, snapshot, component_updates_available, force=False):
    """Sets that we just did an update check and found the given snapshot.

    If the given snapshot is different than the last one we saw, refresh the set
    of activated notifications for available updates for any notifications with
    matching conditions.

    You must call Save() to persist these changes or use this as a context
    manager.

    Args:
      snapshot: snapshots.ComponentSnapshot, The latest snapshot available.
      component_updates_available: bool, True if there are updates to components
        we have installed.  False otherwise.
      force: bool, True to force a recalculation of whether there are available
        updates, even if the snapshot revision has not changed.
    """
    if force or self.LastUpdateCheckRevision() != snapshot.revision:
        log.debug('Updating notification cache...')
        current_version = config.INSTALLATION_CONFIG.version
        current_revision = config.INSTALLATION_CONFIG.revision
        activated = []
        possible_notifications = snapshot.sdk_definition.notifications
        for notification in possible_notifications:
            if notification.condition.Matches(current_version, current_revision, component_updates_available):
                log.debug('Activating notification: [%s]', notification.id)
                activated.append(notification)
        self._data.notifications = activated
        self._CleanUpLastNagTimes()
    self._data.last_update_check_time = time.time()
    self._data.last_update_check_revision = snapshot.revision
    self._dirty = True