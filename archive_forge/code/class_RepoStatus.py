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
class RepoStatus:
    """RepoStatus represents an aggregated repo status after deduplication."""

    def __init__(self):
        self.synced = 0
        self.pending = 0
        self.error = 0
        self.stalled = 0
        self.reconciling = 0
        self.total = 0
        self.namespace = ''
        self.name = ''
        self.source = ''
        self.cluster_type = ''