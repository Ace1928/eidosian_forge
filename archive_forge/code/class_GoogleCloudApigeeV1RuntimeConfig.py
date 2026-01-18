from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1RuntimeConfig(_messages.Message):
    """Runtime configuration for the organization. Response for
  GetRuntimeConfig.

  Fields:
    analyticsBucket: Cloud Storage bucket used for uploading Analytics
      records.
    name: Name of the resource in the following format:
      `organizations/{org}/runtimeConfig`.
    tenantProjectId: Output only. Tenant project ID associated with the Apigee
      organization. The tenant project is used to host Google-managed
      resources that are dedicated to this Apigee organization. Clients have
      limited access to resources within the tenant project used to support
      Apigee runtime instances. Access to the tenant project is managed using
      SetSyncAuthorization. It can be empty if the tenant project hasn't been
      created yet.
    traceBucket: Cloud Storage bucket used for uploading Trace records.
  """
    analyticsBucket = _messages.StringField(1)
    name = _messages.StringField(2)
    tenantProjectId = _messages.StringField(3)
    traceBucket = _messages.StringField(4)