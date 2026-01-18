from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceIdentity(_messages.Message):
    """The per-product per-project service identity for a service.   Use this
  field to configure per-product per-project service identity. Example of a
  service identity configuration.      usage:       service_identity:       -
  service_account_parent: "projects/123456789"         display_name: "Cloud
  XXX Service Agent"         description: "Used as the identity of Cloud XXX
  to access resources"

  Fields:
    description: Optional. A user-specified opaque description of the service
      account. Must be less than or equal to 256 UTF-8 bytes.
    displayName: Optional. A user-specified name for the service account. Must
      be less than or equal to 100 UTF-8 bytes.
    serviceAccountParent: A service account project that hosts the service
      accounts.  An example name would be: `projects/123456789`
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    serviceAccountParent = _messages.StringField(3)