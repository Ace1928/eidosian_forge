from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesStartRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesStartRequest object.

  Fields:
    name: Required. Name of the resource.
    startInstanceRequest: A StartInstanceRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    startInstanceRequest = _messages.MessageField('StartInstanceRequest', 2)