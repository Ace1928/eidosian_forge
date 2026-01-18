from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SyncAuthorization(_messages.Message):
    """A GoogleCloudApigeeV1SyncAuthorization object.

  Fields:
    etag: Entity tag (ETag) used for optimistic concurrency control as a way
      to help prevent simultaneous updates from overwriting each other. For
      example, when you call
      [getSyncAuthorization](organizations/getSyncAuthorization) an ETag is
      returned in the response. Pass that ETag when calling the
      [setSyncAuthorization](organizations/setSyncAuthorization) to ensure
      that you are updating the correct version. If you don't pass the ETag in
      the call to `setSyncAuthorization`, then the existing authorization is
      overwritten indiscriminately. **Note**: We strongly recommend that you
      use the ETag in the read-modify-write cycle to avoid race conditions.
    identities: Required. Array of service accounts to grant access to control
      plane resources, each specified using the following format:
      `serviceAccount:` service-account-name. The service-account-name is
      formatted like an email address. For example: `my-synchronizer-manager-
      service_account@my_project_id.iam.gserviceaccount.com` You might specify
      multiple service accounts, for example, if you have multiple
      environments and wish to assign a unique service account to each one.
      The service accounts must have **Apigee Synchronizer Manager** role. See
      also [Create service
      accounts](https://cloud.google.com/apigee/docs/hybrid/latest/sa-
      about#create-the-service-accounts).
  """
    etag = _messages.BytesField(1)
    identities = _messages.StringField(2, repeated=True)