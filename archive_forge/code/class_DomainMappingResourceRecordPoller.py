from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
class DomainMappingResourceRecordPoller(waiter.OperationPoller):
    """Poll for when a DomainMapping first has resourceRecords."""

    def __init__(self, ops):
        self._ops = ops

    def IsDone(self, mapping):
        if getattr(mapping.status, 'resourceRecords', None):
            return True
        conditions = mapping.conditions
        if conditions and conditions['Ready']['status'] is False:
            return True
        return False

    def GetResult(self, mapping):
        return mapping

    def Poll(self, domain_mapping_ref):
        return self._ops.GetDomainMapping(domain_mapping_ref)