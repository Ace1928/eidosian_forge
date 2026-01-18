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
class DescribeResult:
    """DescribeResult represents the result of the describe command."""

    def __init__(self):
        self.detailed_status = []
        self.managed_resources = []

    def AppendDetailedStatus(self, status):
        for i in range(len(self.detailed_status)):
            s = self.detailed_status[i]
            if s.EqualTo(status):
                s.clusters.append(status.clusters[0])
                self.detailed_status[i] = s
                return
        self.detailed_status.append(status)

    def AppendManagedResources(self, resource, membership, status):
        """append a managed resource to the list."""
        if status.lower() != 'all' and resource['status'].lower() != status.lower():
            return
        for i in range(len(self.managed_resources)):
            r = self.managed_resources[i]
            if r.group == resource['group'] and r.kind == resource['kind'] and (r.namespace == resource['namespace']) and (r.name == resource['name']) and (r.status == resource['status']):
                r.clusters.append(membership)
                self.managed_resources[i] = r
                return
        conditions = None
        if 'conditions' in resource:
            conditions = resource['conditions'][:]
        reconcile_condition = utils.GetActuationCondition(resource)
        if reconcile_condition is not None:
            conditions = [] if conditions is None else conditions
            conditions.insert(0, reconcile_condition)
        source_hash = resource.get('sourceHash', '')
        mr = ManagedResource(group=resource['group'], kind=resource['kind'], namespace=resource['namespace'], name=resource['name'], source_hash=source_hash, status=resource.get('status', ''), conditions=conditions, clusters=[membership])
        self.managed_resources.append(mr)