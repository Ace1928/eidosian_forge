from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnthospolicycontrollerstatusPaProjectsFleetConstraintTemplatesGetRequest(_messages.Message):
    """A
  AnthospolicycontrollerstatusPaProjectsFleetConstraintTemplatesGetRequest
  object.

  Fields:
    name: Required. The name of the fleet constraint template to retrieve.
      Format: projects/{project_id}/fleetConstraintTemplates/{constraint_templ
      ate_name}.
  """
    name = _messages.StringField(1, required=True)