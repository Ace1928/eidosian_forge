from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UndeleteCustomClassRequest(_messages.Message):
    """Request message for the UndeleteCustomClass method.

  Fields:
    etag: This checksum is computed by the server based on the value of other
      fields. This may be sent on update, undelete, and delete requests to
      ensure the client has an up-to-date value before proceeding.
    name: Required. The name of the CustomClass to undelete. Format:
      `projects/{project}/locations/{location}/customClasses/{custom_class}`
    validateOnly: If set, validate the request and preview the undeleted
      CustomClass, but do not actually undelete it.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)