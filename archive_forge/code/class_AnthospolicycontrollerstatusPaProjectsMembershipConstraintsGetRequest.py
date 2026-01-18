from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnthospolicycontrollerstatusPaProjectsMembershipConstraintsGetRequest(_messages.Message):
    """A AnthospolicycontrollerstatusPaProjectsMembershipConstraintsGetRequest
  object.

  Fields:
    name: Required. The name of the membership constraint to retrieve. Format:
      projects/{project_id}/membershipConstraints/{constraint_template_name}/{
      constraint_name}/{membership_uuid}.
  """
    name = _messages.StringField(1, required=True)