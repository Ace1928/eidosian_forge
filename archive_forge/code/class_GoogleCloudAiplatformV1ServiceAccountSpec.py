from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ServiceAccountSpec(_messages.Message):
    """Configuration for the use of custom service account to run the
  workloads.

  Fields:
    enableCustomServiceAccount: Required. If true, custom user-managed service
      account is enforced to run any workloads (for example, Vertex Jobs) on
      the resource. Otherwise, uses the [Vertex AI Custom Code Service
      Agent](https://cloud.google.com/vertex-ai/docs/general/access-
      control#service-agents).
    serviceAccount: Optional. Default service account that this
      PersistentResource's workloads run as. The workloads include: * Any
      runtime specified via `ResourceRuntimeSpec` on creation time, for
      example, Ray. * Jobs submitted to PersistentResource, if no other
      service account specified in the job specs. Only works when custom
      service account is enabled and users have the
      `iam.serviceAccounts.actAs` permission on this service account. Required
      if any containers are specified in `ResourceRuntimeSpec`.
  """
    enableCustomServiceAccount = _messages.BooleanField(1)
    serviceAccount = _messages.StringField(2)