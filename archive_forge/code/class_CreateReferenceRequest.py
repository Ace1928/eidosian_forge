from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateReferenceRequest(_messages.Message):
    """The CreateReferenceRequest request.

  Fields:
    parent: Required. The parent resource name (target_resource of this
      reference). For example: `//targetservice.googleapis.com/projects/{my-
      project}/locations/{location}/instances/{my-instance}`.
    reference: Required. The reference to be created.
    referenceId: The unique id of this resource. Must be unique within a scope
      of a target resource, but does not have to be globally unique. Reference
      ID is part of resource name of the reference. Resource name is generated
      in the following way: {parent}/references/{reference_id}. Reference ID
      field is currently required but id auto generation might be added in the
      future. It can be any arbitrary string, either GUID or any other string,
      however CLHs can use preprocess callbacks to perform a custom
      validation.
    requestId: Optional. Request ID is an idempotency ID of the request. It
      must be a valid UUID. Zero UUID (00000000-0000-0000-0000-000000000000)
      is not supported.
  """
    parent = _messages.StringField(1)
    reference = _messages.MessageField('Reference', 2)
    referenceId = _messages.StringField(3)
    requestId = _messages.StringField(4)