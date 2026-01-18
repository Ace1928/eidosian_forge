from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ServiceAccountSpec(_messages.Message):
    """Configuration for the use of custom service account to run the
  workloads.

  Fields:
    enableCustomServiceAccount: Required. If true, custom user-managed service
      account is enforced to run any workloads (for example, Vertex Jobs) on
      the resource. Otherwise, uses the [Vertex AI Custom Code Service
      Agent](https://cloud.google.com/vertex-ai/docs/general/access-
      control#service-agents).
    serviceAccount: Optional. Required when all below conditions are met *
      `enable_custom_service_account` is true; * any runtime is specified via
      `ResourceRuntimeSpec` on creation time, for example, Ray The users must
      have `iam.serviceAccounts.actAs` permission on this service account and
      then the specified runtime containers will run as it. Do not set this
      field if you want to submit jobs using custom service account to this
      PersistentResource after creation, but only specify the
      `service_account` inside the job.
  """
    enableCustomServiceAccount = _messages.BooleanField(1)
    serviceAccount = _messages.StringField(2)