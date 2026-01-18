from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityStatsQueryTabularStatsRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityStatsQueryTabularStatsRequest
  object.

  Fields:
    googleCloudApigeeV1QueryTabularStatsRequest: A
      GoogleCloudApigeeV1QueryTabularStatsRequest resource to be passed as the
      request body.
    orgenv: Required. Should be of the form organizations//environments/.
  """
    googleCloudApigeeV1QueryTabularStatsRequest = _messages.MessageField('GoogleCloudApigeeV1QueryTabularStatsRequest', 1)
    orgenv = _messages.StringField(2, required=True)