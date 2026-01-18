from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
def CreateDiff(self, latest_snapshot, platform_filter=None):
    """Creates a ComponentSnapshotDiff based on this snapshot and the given one.

    Args:
      latest_snapshot: ComponentSnapshot, The latest state of the world that we
        want to compare to.
      platform_filter: platforms.Platform, A platform that components must
        match in order to be considered for any operations.

    Returns:
      A ComponentSnapshotDiff object.
    """
    return ComponentSnapshotDiff(self, latest_snapshot, platform_filter=platform_filter)