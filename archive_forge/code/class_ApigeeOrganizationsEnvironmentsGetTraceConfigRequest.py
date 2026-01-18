from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsGetTraceConfigRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsGetTraceConfigRequest object.

  Fields:
    name: Required. Name of the trace configuration. Use the following
      structure in your request: "organizations/*/environments/*/traceConfig".
  """
    name = _messages.StringField(1, required=True)