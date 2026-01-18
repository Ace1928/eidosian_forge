from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsCreateRequest
  object.

  Fields:
    googleCloudApigeeV1DebugSession: A GoogleCloudApigeeV1DebugSession
      resource to be passed as the request body.
    parent: Required. The resource name of the API Proxy revision deployment
      for which to create the DebugSession. Must be of the form `organizations
      /{organization}/environments/{environment}/apis/{api}/revisions/{revisio
      n}`.
    timeout: Optional. The time in seconds after which this DebugSession
      should end. A timeout specified in DebugSession will overwrite this
      value.
  """
    googleCloudApigeeV1DebugSession = _messages.MessageField('GoogleCloudApigeeV1DebugSession', 1)
    parent = _messages.StringField(2, required=True)
    timeout = _messages.IntegerField(3)