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
class RepoResourceGroupPair:
    """RepoResourceGroupPair represents a RootSync|RepoSync and a ResourceGroup pair."""

    def __init__(self, repo, rg, cluster_type):
        self.repo = repo
        self.rg = rg
        self.cluster_type = cluster_type