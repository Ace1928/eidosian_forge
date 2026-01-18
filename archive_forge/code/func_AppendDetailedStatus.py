from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def AppendDetailedStatus(self, status):
    for i in range(len(self.detailed_status)):
        s = self.detailed_status[i]
        if s.EqualTo(status):
            s.clusters.append(status.clusters[0])
            self.detailed_status[i] = s
            return
    self.detailed_status.append(status)