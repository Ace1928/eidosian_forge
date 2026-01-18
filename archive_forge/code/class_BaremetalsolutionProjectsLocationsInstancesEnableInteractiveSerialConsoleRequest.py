from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesEnableInteractiveSerialConsoleRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesEnableInteractiveSerialCons
  oleRequest object.

  Fields:
    enableInteractiveSerialConsoleRequest: A
      EnableInteractiveSerialConsoleRequest resource to be passed as the
      request body.
    name: Required. Name of the resource.
  """
    enableInteractiveSerialConsoleRequest = _messages.MessageField('EnableInteractiveSerialConsoleRequest', 1)
    name = _messages.StringField(2, required=True)