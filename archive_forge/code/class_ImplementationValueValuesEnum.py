from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImplementationValueValuesEnum(_messages.Enum):
    """Output only. Implementation of managed control plane.

    Values:
      IMPLEMENTATION_UNSPECIFIED: Unspecified
      ISTIOD: A Google build of istiod is used for the managed control plane.
      TRAFFIC_DIRECTOR: Traffic director is used for the managed control
        plane.
      UPDATING: The control plane implementation is being updated.
    """
    IMPLEMENTATION_UNSPECIFIED = 0
    ISTIOD = 1
    TRAFFIC_DIRECTOR = 2
    UPDATING = 3