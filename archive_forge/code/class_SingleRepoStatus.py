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
class SingleRepoStatus:
    """SingleRepoStatus represents a single repo status on a single cluster."""

    def __init__(self, status, errors, commit):
        self.status = status
        self.errors = errors
        self.commit = commit

    def GetStatus(self):
        return self.status

    def GetErrors(self):
        return self.errors

    def GetCommit(self):
        return self.commit