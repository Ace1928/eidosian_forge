from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
def _ClearMutualExclusiveBackendCapacityThresholds(backend):
    """Initialize the backend's mutually exclusive capacity thresholds."""
    backend.maxRate = None
    backend.maxRatePerInstance = None
    backend.maxConnections = None
    backend.maxConnectionsPerInstance = None
    backend.maxRatePerEndpoint = None
    backend.maxConnectionsPerEndpoint = None