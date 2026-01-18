from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionInstanceTemplatesGetRequest(_messages.Message):
    """A ComputeRegionInstanceTemplatesGetRequest object.

  Enums:
    ViewValueValuesEnum: View of the instance template.

  Fields:
    instanceTemplate: The name of the instance template.
    project: Project ID for this request.
    region: The name of the region for this request.
    view: View of the instance template.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """View of the instance template.

    Values:
      BASIC: Include everything except Partner Metadata.
      FULL: Include everything.
      INSTANCE_VIEW_UNSPECIFIED: The default / unset value. The API will
        default to the BASIC view.
    """
        BASIC = 0
        FULL = 1
        INSTANCE_VIEW_UNSPECIFIED = 2
    instanceTemplate = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)