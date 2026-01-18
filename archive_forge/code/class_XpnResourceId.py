from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class XpnResourceId(_messages.Message):
    """Service resource (a.k.a service project) ID.

  Enums:
    TypeValueValuesEnum: The type of the service resource.

  Fields:
    id: The ID of the service resource. In the case of projects, this field
      supports project id (e.g., my-project-123) and project number (e.g.
      12345678).
    type: The type of the service resource.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the service resource.

    Values:
      PROJECT: <no description>
      XPN_RESOURCE_TYPE_UNSPECIFIED: <no description>
    """
        PROJECT = 0
        XPN_RESOURCE_TYPE_UNSPECIFIED = 1
    id = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)