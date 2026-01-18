from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetAnalyzeMoveRequest(_messages.Message):
    """A CloudassetAnalyzeMoveRequest object.

  Enums:
    ViewValueValuesEnum: Analysis view indicating what information should be
      included in the analysis response. If unspecified, the default view is
      FULL.

  Fields:
    destinationParent: Required. Name of the Google Cloud folder or
      organization to reparent the target resource. The analysis will be
      performed against hypothetically moving the resource to this specified
      desitination parent. This can only be a folder number (such as
      "folders/123") or an organization number (such as "organizations/123").
    resource: Required. Name of the resource to perform the analysis against.
      Only Google Cloud projects are supported as of today. Hence, this can
      only be a project ID (such as "projects/my-project-id") or a project
      number (such as "projects/12345").
    view: Analysis view indicating what information should be included in the
      analysis response. If unspecified, the default view is FULL.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Analysis view indicating what information should be included in the
    analysis response. If unspecified, the default view is FULL.

    Values:
      ANALYSIS_VIEW_UNSPECIFIED: The default/unset value. The API will default
        to the FULL view.
      FULL: Full analysis including all level of impacts of the specified
        resource move.
      BASIC: Basic analysis only including blockers which will prevent the
        specified resource move at runtime.
    """
        ANALYSIS_VIEW_UNSPECIFIED = 0
        FULL = 1
        BASIC = 2
    destinationParent = _messages.StringField(1)
    resource = _messages.StringField(2, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 3)