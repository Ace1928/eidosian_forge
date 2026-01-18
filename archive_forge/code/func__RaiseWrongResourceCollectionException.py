from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.operations import flags
from googlecloudsdk.core import resources
def _RaiseWrongResourceCollectionException(self, got, path):
    expected_collections = ['compute.instances', 'compute.globalOperations', 'compute.regionOperations', 'compute.zoneOperations']
    raise resources.WrongResourceCollectionException(expected=','.join(expected_collections), got=got, path=path)