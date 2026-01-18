from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsGetRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsGetRequest
  object.

  Fields:
    name: Required. The name of the debug session to retrieve. Must be of the
      form: `organizations/{organization}/environments/{environment}/apis/{api
      }/revisions/{revision}/debugsessions/{session}`.
  """
    name = _messages.StringField(1, required=True)