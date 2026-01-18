from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsResourceValueConfigsBatchCreateRequest(_messages.Message):
    """A SecuritycenterOrganizationsResourceValueConfigsBatchCreateRequest
  object.

  Fields:
    batchCreateResourceValueConfigsRequest: A
      BatchCreateResourceValueConfigsRequest resource to be passed as the
      request body.
    parent: Required. Resource name of the new ResourceValueConfig's parent.
      The parent field in the CreateResourceValueConfigRequest messages must
      either be empty or match this field.
  """
    batchCreateResourceValueConfigsRequest = _messages.MessageField('BatchCreateResourceValueConfigsRequest', 1)
    parent = _messages.StringField(2, required=True)