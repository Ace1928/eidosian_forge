from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudBuildMembershipSpec(_messages.Message):
    """**Cloud Build**: Configurations for each Cloud Build enabled cluster.

  Enums:
    SecurityPolicyValueValuesEnum: Whether it is allowed to run the privileged
      builds on the cluster or not.

  Fields:
    securityPolicy: Whether it is allowed to run the privileged builds on the
      cluster or not.
    version: Version of the cloud build software on the cluster.
  """

    class SecurityPolicyValueValuesEnum(_messages.Enum):
        """Whether it is allowed to run the privileged builds on the cluster or
    not.

    Values:
      SECURITY_POLICY_UNSPECIFIED: Unspecified policy
      NON_PRIVILEGED: Privileged build pods are disallowed
      PRIVILEGED: Privileged build pods are allowed
    """
        SECURITY_POLICY_UNSPECIFIED = 0
        NON_PRIVILEGED = 1
        PRIVILEGED = 2
    securityPolicy = _messages.EnumField('SecurityPolicyValueValuesEnum', 1)
    version = _messages.StringField(2)