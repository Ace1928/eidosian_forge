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
def AllDiffs(self):
    """Gets all ComponentDiffs for this snapshot comparison.

    Returns:
      The list of all ComponentDiffs between the snapshots.
    """
    return self._FilterDiffs(None)