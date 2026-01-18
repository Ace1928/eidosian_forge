from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceAgentAuthValueValuesEnum(_messages.Enum):
    """Optional. Indicate the auth token type generated from the [Diglogflow
    service agent](https://cloud.google.com/iam/docs/service-
    agents#dialogflow-service-agent). The generated token is sent in the
    Authorization header.

    Values:
      SERVICE_AGENT_AUTH_UNSPECIFIED: Service agent auth type unspecified.
        Default to ID_TOKEN.
      NONE: No token used.
      ID_TOKEN: Use [ID
        token](https://cloud.google.com/docs/authentication/token-types#id)
        generated from service agent. This can be used to access Cloud
        Function and Cloud Run after you grant Invoker role to `service-@gcp-
        sa-dialogflow.iam.gserviceaccount.com`.
      ACCESS_TOKEN: Use [access
        token](https://cloud.google.com/docs/authentication/token-
        types#access) generated from service agent. This can be used to access
        other Google Cloud APIs after you grant required roles to
        `service-@gcp-sa-dialogflow.iam.gserviceaccount.com`.
    """
    SERVICE_AGENT_AUTH_UNSPECIFIED = 0
    NONE = 1
    ID_TOKEN = 2
    ACCESS_TOKEN = 3