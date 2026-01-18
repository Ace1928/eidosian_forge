from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WorkflowexecutionsProjectsLocationsWorkflowsExecutionsGetRequest(_messages.Message):
    """A WorkflowexecutionsProjectsLocationsWorkflowsExecutionsGetRequest
  object.

  Enums:
    ViewValueValuesEnum: Optional. A view defining which fields should be
      filled in the returned execution. The API will default to the FULL view.

  Fields:
    name: Required. Name of the execution to be retrieved. Format: projects/{p
      roject}/locations/{location}/workflows/{workflow}/executions/{execution}
    view: Optional. A view defining which fields should be filled in the
      returned execution. The API will default to the FULL view.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. A view defining which fields should be filled in the
    returned execution. The API will default to the FULL view.

    Values:
      EXECUTION_VIEW_UNSPECIFIED: The default / unset value.
      BASIC: Includes only basic metadata about the execution. Following
        fields are returned: name, start_time, end_time, state and
        workflow_revision_id.
      FULL: Includes all data.
    """
        EXECUTION_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)