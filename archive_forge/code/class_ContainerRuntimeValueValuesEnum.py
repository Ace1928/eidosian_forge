from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerRuntimeValueValuesEnum(_messages.Enum):
    """Specifies which container runtime will be used.

    Values:
      CONTAINER_RUNTIME_UNSPECIFIED: No container runtime selected.
      CONTAINERD: Containerd runtime.
    """
    CONTAINER_RUNTIME_UNSPECIFIED = 0
    CONTAINERD = 1