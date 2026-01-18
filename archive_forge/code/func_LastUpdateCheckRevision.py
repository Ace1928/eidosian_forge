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
def LastUpdateCheckRevision(self):
    """Gets the revision of the snapshot from the last update check.

    Returns:
      long, The revision of the last checked snapshot.  This is a long int but
        formatted as an actual date in seconds (i.e 20151009132504). It is *NOT*
        seconds since the epoch.
    """
    return self._data.last_update_check_revision