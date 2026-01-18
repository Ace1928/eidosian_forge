from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnthospolicycontrollerstatusPaProjectsFleetConstraintsGetRequest(_messages.Message):
    """A AnthospolicycontrollerstatusPaProjectsFleetConstraintsGetRequest
  object.

  Fields:
    name: Required. The name of the fleet constraint to retrieve. Format: proj
      ects/{project_id}/fleetConstraints/{constraint_template_name}/{constrain
      t_name}.
  """
    name = _messages.StringField(1, required=True)