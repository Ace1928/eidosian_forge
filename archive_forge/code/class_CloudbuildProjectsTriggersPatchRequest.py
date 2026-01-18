from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsTriggersPatchRequest(_messages.Message):
    """A CloudbuildProjectsTriggersPatchRequest object.

  Fields:
    buildTrigger: A BuildTrigger resource to be passed as the request body.
    projectId: Required. ID of the project that owns the trigger.
    triggerId: Required. ID of the `BuildTrigger` to update.
    updateMask: Update mask for the resource. If this is set, the server will
      only update the fields specified in the field mask. Otherwise, a full
      update of the mutable resource fields will be performed.
  """
    buildTrigger = _messages.MessageField('BuildTrigger', 1)
    projectId = _messages.StringField(2, required=True)
    triggerId = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)