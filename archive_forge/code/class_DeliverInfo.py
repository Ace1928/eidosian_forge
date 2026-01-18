from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeliverInfo(_messages.Message):
    """Details of the final state "deliver" and associated resource.

  Enums:
    TargetValueValuesEnum: Target type where the packet is delivered to.

  Fields:
    ipAddress: IP address of the target (if applicable).
    resourceUri: URI of the resource that the packet is delivered to.
    target: Target type where the packet is delivered to.
  """

    class TargetValueValuesEnum(_messages.Enum):
        """Target type where the packet is delivered to.

    Values:
      TARGET_UNSPECIFIED: Target not specified.
      INSTANCE: Target is a Compute Engine instance.
      INTERNET: Target is the internet.
      GOOGLE_API: Target is a Google API.
      GKE_MASTER: Target is a Google Kubernetes Engine cluster master.
      CLOUD_SQL_INSTANCE: Target is a Cloud SQL instance.
      PSC_PUBLISHED_SERVICE: Target is a published service that uses [Private
        Service Connect](https://cloud.google.com/vpc/docs/configure-private-
        service-connect-services).
      PSC_GOOGLE_API: Target is all Google APIs that use [Private Service
        Connect](https://cloud.google.com/vpc/docs/configure-private-service-
        connect-apis).
      PSC_VPC_SC: Target is a VPC-SC that uses [Private Service
        Connect](https://cloud.google.com/vpc/docs/configure-private-service-
        connect-apis).
      SERVERLESS_NEG: Target is a serverless network endpoint group.
      STORAGE_BUCKET: Target is a Cloud Storage bucket.
      PRIVATE_NETWORK: Target is a private network. Used only for return
        traces.
      CLOUD_FUNCTION: Target is a Cloud Function. Used only for return traces.
      APP_ENGINE_VERSION: Target is a App Engine service version. Used only
        for return traces.
      CLOUD_RUN_REVISION: Target is a Cloud Run revision. Used only for return
        traces.
    """
        TARGET_UNSPECIFIED = 0
        INSTANCE = 1
        INTERNET = 2
        GOOGLE_API = 3
        GKE_MASTER = 4
        CLOUD_SQL_INSTANCE = 5
        PSC_PUBLISHED_SERVICE = 6
        PSC_GOOGLE_API = 7
        PSC_VPC_SC = 8
        SERVERLESS_NEG = 9
        STORAGE_BUCKET = 10
        PRIVATE_NETWORK = 11
        CLOUD_FUNCTION = 12
        APP_ENGINE_VERSION = 13
        CLOUD_RUN_REVISION = 14
    ipAddress = _messages.StringField(1)
    resourceUri = _messages.StringField(2)
    target = _messages.EnumField('TargetValueValuesEnum', 3)