from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsTraceConfigOverridesDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsTraceConfigOverridesDeleteRequest
  object.

  Fields:
    name: Required. Name of the trace configuration override. Use the
      following structure in your request:
      "organizations/*/environments/*/traceConfig/overrides/*".
  """
    name = _messages.StringField(1, required=True)