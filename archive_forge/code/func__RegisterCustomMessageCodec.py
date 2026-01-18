from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from apitools.base.py import http_wrapper
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.spanner.sql import QueryHasDml
def _RegisterCustomMessageCodec(msgs):
    """Register custom message code.

  Args:
    msgs: Spanner v1 messages.
  """

    def _ToJson(msg):
        return extra_types.JsonProtoEncoder(extra_types.JsonArray(entries=msg.entry))

    def _FromJson(data):
        return msgs.ResultSet.RowsValueListEntry(entry=extra_types.JsonProtoDecoder(data).entries)
    encoding.RegisterCustomMessageCodec(encoder=_ToJson, decoder=_FromJson)(msgs.ResultSet.RowsValueListEntry)