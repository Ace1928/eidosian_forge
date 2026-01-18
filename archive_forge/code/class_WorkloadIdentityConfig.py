from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadIdentityConfig(_messages.Message):
    """Configuration for the use of Kubernetes Service Accounts in GCP IAM
  policies.

  Fields:
    workloadPool: The workload pool to attach all Kubernetes service accounts
      to.
  """
    workloadPool = _messages.StringField(1)