from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisVersionsPatchRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisVersionsPatchRequest object.

  Fields:
    allowMissing: If set to true, and the version is not found, a new version
      will be created. In this situation, `update_mask` is ignored.
    apiVersion: A ApiVersion resource to be passed as the request body.
    name: Resource name.
    updateMask: The list of fields to be updated. If omitted, all fields are
      updated that are set in the request message (fields set to default
      values are ignored). If an asterisk "*" is specified, all fields are
      updated, including fields that are unspecified/default in the request.
  """
    allowMissing = _messages.BooleanField(1)
    apiVersion = _messages.MessageField('ApiVersion', 2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)