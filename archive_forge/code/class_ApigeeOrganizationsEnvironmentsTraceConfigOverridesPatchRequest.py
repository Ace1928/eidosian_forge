from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsTraceConfigOverridesPatchRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsTraceConfigOverridesPatchRequest
  object.

  Fields:
    googleCloudApigeeV1TraceConfigOverride: A
      GoogleCloudApigeeV1TraceConfigOverride resource to be passed as the
      request body.
    name: Required. Name of the trace configuration override. Use the
      following structure in your request:
      "organizations/*/environments/*/traceConfig/overrides/*".
    updateMask: List of fields to be updated.
  """
    googleCloudApigeeV1TraceConfigOverride = _messages.MessageField('GoogleCloudApigeeV1TraceConfigOverride', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)