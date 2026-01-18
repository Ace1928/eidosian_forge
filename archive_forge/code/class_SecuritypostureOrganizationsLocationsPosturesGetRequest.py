from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritypostureOrganizationsLocationsPosturesGetRequest(_messages.Message):
    """A SecuritypostureOrganizationsLocationsPosturesGetRequest object.

  Fields:
    name: Required. Name of the resource.
    revisionId: Optional. Posture revision which needs to be retrieved.
  """
    name = _messages.StringField(1, required=True)
    revisionId = _messages.StringField(2)