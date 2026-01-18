from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaUpdateInstanceRequest(_messages.Message):
    """The request used for `UpdateInstance`.

  Fields:
    instance: Specifies the instance to update.
    loggingEnabled: Deprecated, use instance.logging_enabled instead. Whether
      to enable Stackdriver logging for this instance.
    name: Deprecated, use instance.Name instead. Name of the instance to
      update. Format: `projects/[PROJECT_ID]/instances/[INSTANCE_ID]`.
    updateMask: The update mask applies to instance. For the `FieldMask`
      definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask If an empty update_mask
      is provided, only the non-default valued field in the worker pool field
      will be updated. Note that in order to update a field to the default
      value (zero, false, empty string) an explicit update_mask must be
      provided.
  """
    instance = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaInstance', 1)
    loggingEnabled = _messages.BooleanField(2)
    name = _messages.StringField(3)
    updateMask = _messages.StringField(4)