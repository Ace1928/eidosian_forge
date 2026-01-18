from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyControllerPolicyControllerDeploymentConfig(_messages.Message):
    """Deployment-specific configuration.

  Enums:
    PodAffinityValueValuesEnum: Pod affinity configuration.

  Fields:
    containerResources: Container resource requirements.
    podAffinity: Pod affinity configuration.
    podAntiAffinity: Pod anti-affinity enablement. Deprecated: use
      `pod_affinity` instead.
    podTolerations: Pod tolerations of node taints.
    replicaCount: Pod replica count.
  """

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
    containerResources = _messages.MessageField('PolicyControllerResourceRequirements', 1)
    podAffinity = _messages.EnumField('PodAffinityValueValuesEnum', 2)
    podAntiAffinity = _messages.BooleanField(3)
    podTolerations = _messages.MessageField('PolicyControllerToleration', 4, repeated=True)
    replicaCount = _messages.IntegerField(5)