from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvaluationModeValueValuesEnum(_messages.Enum):
    """Mode of operation for binauthz policy evaluation. If unspecified,
    defaults to DISABLED.

    Values:
      EVALUATION_MODE_UNSPECIFIED: Default value
      DISABLED: Disable BinaryAuthorization
      PROJECT_SINGLETON_POLICY_ENFORCE: Enforce Kubernetes admission requests
        with BinaryAuthorization using the project's singleton policy.
    """
    EVALUATION_MODE_UNSPECIFIED = 0
    DISABLED = 1
    PROJECT_SINGLETON_POLICY_ENFORCE = 2