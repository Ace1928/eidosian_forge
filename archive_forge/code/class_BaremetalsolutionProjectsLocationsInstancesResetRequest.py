from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesResetRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesResetRequest object.

  Fields:
    name: Required. Name of the resource.
    resetInstanceRequest: A ResetInstanceRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    resetInstanceRequest = _messages.MessageField('ResetInstanceRequest', 2)