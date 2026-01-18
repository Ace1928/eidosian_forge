from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsUpdateTraceConfigRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsUpdateTraceConfigRequest object.

  Fields:
    googleCloudApigeeV1TraceConfig: A GoogleCloudApigeeV1TraceConfig resource
      to be passed as the request body.
    name: Required. Name of the trace configuration. Use the following
      structure in your request: "organizations/*/environments/*/traceConfig".
    updateMask: List of fields to be updated.
  """
    googleCloudApigeeV1TraceConfig = _messages.MessageField('GoogleCloudApigeeV1TraceConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)