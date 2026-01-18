from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomMirroringProfile(_messages.Message):
    """CustomMirroringProfile defines an action for mirroring traffic to a
  collector's EndpointGroup

  Fields:
    mirroringEndpointGroup: Required. The MirroringEndpointGroup to which
      traffic associated with the SP should be mirrored.
  """
    mirroringEndpointGroup = _messages.StringField(1)