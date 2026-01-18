from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstancesGetRequest(_messages.Message):
    """A ComputeInstancesGetRequest object.

  Enums:
    ViewValueValuesEnum: View of the instance.

  Fields:
    instance: Name of the instance resource to return.
    project: Project ID for this request.
    view: View of the instance.
    zone: The name of the zone for this request.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """View of the instance.

    Values:
      BASIC: Include everything except Partner Metadata.
      FULL: Include everything.
      INSTANCE_VIEW_UNSPECIFIED: The default / unset value. The API will
        default to the BASIC view.
    """
        BASIC = 0
        FULL = 1
        INSTANCE_VIEW_UNSPECIFIED = 2
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 3)
    zone = _messages.StringField(4, required=True)