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
class ManagedResource:
    """ManagedResource represent a managed resource across multiple clusters."""

    def __init__(self, group='', kind='', namespace='', name='', source_hash='', status='', conditions=None, clusters=None):
        if not conditions:
            self.conditions = None
        else:
            messages = []
            for condition in conditions:
                messages.append(condition['message'])
            self.conditions = messages
        self.group = group
        self.kind = kind
        self.namespace = namespace
        self.name = name
        self.status = status
        self.source_hash = source_hash
        self.clusters = clusters