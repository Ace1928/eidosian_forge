from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcaccessProjectsLocationsConnectorsPatchRequest(_messages.Message):
    """A VpcaccessProjectsLocationsConnectorsPatchRequest object.

  Fields:
    connector: A Connector resource to be passed as the request body.
    name: The resource name in the format
      `projects/*/locations/*/connectors/*`.
    updateMask: The fields to update on the entry group. If absent or empty,
      all modifiable fields are updated.
  """
    connector = _messages.MessageField('Connector', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)