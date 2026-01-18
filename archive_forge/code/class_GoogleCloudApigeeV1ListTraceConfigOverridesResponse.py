from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListTraceConfigOverridesResponse(_messages.Message):
    """Response for ListTraceConfigOverrides.

  Fields:
    nextPageToken: Token value that can be passed as `page_token` to retrieve
      the next page of content.
    traceConfigOverrides: List all trace configuration overrides in an
      environment.
  """
    nextPageToken = _messages.StringField(1)
    traceConfigOverrides = _messages.MessageField('GoogleCloudApigeeV1TraceConfigOverride', 2, repeated=True)