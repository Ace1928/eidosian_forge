from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritypostureOrganizationsLocationsPostureTemplatesGetRequest(_messages.Message):
    """A SecuritypostureOrganizationsLocationsPostureTemplatesGetRequest
  object.

  Fields:
    name: Required. Name of the resource.
    revisionId: Optional. Specific revision_id of a Posture Template.
      PostureTemplate revision_id which needs to be retrieved.
  """
    name = _messages.StringField(1, required=True)
    revisionId = _messages.StringField(2)