from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SingleAttributeSelector(_messages.Message):
    """Matches a single attribute.

  Fields:
    attribute: Required. The attribute key that will be matched. The following
      attributes are supported: - `attached_service_account` matches workloads
      with the references Google Cloud service account attached. The service
      account should be referenced using its either its email address
      (example: `service-account-id@project-id.iam.gserviceaccount.com`) or
      unique ID (example: `123456789012345678901`). Service account email
      addresses can be reused over time. You should use the service account's
      unique ID if you don't want to match a service account that is deleted,
      and then a new service account is created with the same name.
    value: Required. The value that should exactly match the attribute of the
      workload.
  """
    attribute = _messages.StringField(1)
    value = _messages.StringField(2)