from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsGetRequest(_messages.Message):
    """A
  TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsGetRequest
  object.

  Enums:
    ViewValueValuesEnum: Optional. Defines the type of view of the blueprint.
      When field is not present BLUEPRINT_VIEW_BASIC is considered as default.

  Fields:
    name: Required. The name of the blueprint. Case 1: If the name provided in
      the request is {blueprint_id}@{revision_id}, then the revision with
      revision_id will be returned. Case 2: If the name provided in the
      request is {blueprint}, then the current state of the blueprint is
      returned.
    view: Optional. Defines the type of view of the blueprint. When field is
      not present BLUEPRINT_VIEW_BASIC is considered as default.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. Defines the type of view of the blueprint. When field is not
    present BLUEPRINT_VIEW_BASIC is considered as default.

    Values:
      BLUEPRINT_VIEW_UNSPECIFIED: Unspecified enum value.
      BLUEPRINT_VIEW_BASIC: View which only contains metadata.
      BLUEPRINT_VIEW_FULL: View which contains metadata and files it
        encapsulates.
    """
        BLUEPRINT_VIEW_UNSPECIFIED = 0
        BLUEPRINT_VIEW_BASIC = 1
        BLUEPRINT_VIEW_FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)