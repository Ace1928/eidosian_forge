from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesDisableInteractiveSerialConsoleRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesDisableInteractiveSerialCon
  soleRequest object.

  Fields:
    disableInteractiveSerialConsoleRequest: A
      DisableInteractiveSerialConsoleRequest resource to be passed as the
      request body.
    name: Required. Name of the resource.
  """
    disableInteractiveSerialConsoleRequest = _messages.MessageField('DisableInteractiveSerialConsoleRequest', 1)
    name = _messages.StringField(2, required=True)