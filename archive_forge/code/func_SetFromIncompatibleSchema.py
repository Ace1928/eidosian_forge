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
def SetFromIncompatibleSchema(self):
    """Sets that we just did an update check and found a new schema version.

    An incompatible schema version means there are definitely updates available
    but we can't read the notifications to correctly notify the user.  This will
    install a default notification for the incompatible schema.

    You must call Save() to persist these changes or use this as a context
    manager.
    """
    log.debug('Incompatible schema found.  Activating default notification.')
    notification_spec = schemas.NotificationSpec(id='incompatible', condition=schemas.Condition(None, None, None, None, False), trigger=schemas.Trigger(frequency=604800, command_regex=None), notification=schemas.Notification(None, None, None))
    self._data.notifications = [notification_spec]
    self._CleanUpLastNagTimes()
    self._data.last_update_check_time = time.time()
    self._data.last_update_check_revision = 0
    self._dirty = True