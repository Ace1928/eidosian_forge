from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PodAffinityValueValuesEnum(_messages.Enum):
    """Pod affinity configuration.

    Values:
      AFFINITY_UNSPECIFIED: No affinity configuration has been specified.
      NO_AFFINITY: Affinity configurations will be removed from the
        deployment.
      ANTI_AFFINITY: Anti-affinity configuration will be applied to this
        deployment. Default for admissions deployment.
    """
    AFFINITY_UNSPECIFIED = 0
    NO_AFFINITY = 1
    ANTI_AFFINITY = 2