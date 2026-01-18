from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import resources
def GetOperationIdFromArg(operation):
    if not operation:
        return None
    operation_ref = resources.REGISTRY.Parse(operation, collection='servicemanagement.operations')
    return operation_ref.operationsId