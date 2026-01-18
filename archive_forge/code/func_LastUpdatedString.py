from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
def LastUpdatedString(self):
    try:
        last_updated = config.InstallationConfig.ParseRevision(self.revision)
        return time.strftime('%Y/%m/%d', last_updated)
    except ValueError:
        return 'Unknown'