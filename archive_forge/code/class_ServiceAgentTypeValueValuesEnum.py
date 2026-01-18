from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceAgentTypeValueValuesEnum(_messages.Enum):
    """Optional. Service agent type used during data sync. By default, the
    Vertex AI Service Agent is used. When using an IAM Policy to isolate this
    FeatureView within a project, a separate service account should be
    provisioned by setting this field to `SERVICE_AGENT_TYPE_FEATURE_VIEW`.
    This will generate a separate service account to access the BigQuery
    source table.

    Values:
      SERVICE_AGENT_TYPE_UNSPECIFIED: By default, the project-level Vertex AI
        Service Agent is enabled.
      SERVICE_AGENT_TYPE_PROJECT: Indicates the project-level Vertex AI
        Service Agent (https://cloud.google.com/vertex-ai/docs/general/access-
        control#service-agents) will be used during sync jobs.
      SERVICE_AGENT_TYPE_FEATURE_VIEW: Enable a FeatureView service account to
        be created by Vertex AI and output in the field
        `service_account_email`. This service account will be used to read
        from the source BigQuery table during sync.
    """
    SERVICE_AGENT_TYPE_UNSPECIFIED = 0
    SERVICE_AGENT_TYPE_PROJECT = 1
    SERVICE_AGENT_TYPE_FEATURE_VIEW = 2