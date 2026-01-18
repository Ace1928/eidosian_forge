from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsTriggersPatchRequest(_messages.Message):
    """A EventarcProjectsLocationsTriggersPatchRequest object.

  Fields:
    allowMissing: If set to true, and the trigger is not found, a new trigger
      will be created. In this situation, `update_mask` is ignored.
    name: Required. The resource name of the trigger. Must be unique within
      the location of the project and must be in
      `projects/{project}/locations/{location}/triggers/{trigger}` format.
    trigger: A Trigger resource to be passed as the request body.
    updateMask: The fields to be updated; only fields explicitly provided are
      updated. If no field mask is provided, all provided fields in the
      request are updated. To update all fields, provide a field mask of "*".
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not post it.
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    trigger = _messages.MessageField('Trigger', 3)
    updateMask = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)