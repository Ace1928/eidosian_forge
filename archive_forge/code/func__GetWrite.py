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
@classmethod
def _GetWrite(cls, table, data):
    """Constructs Write object, which is needed for insert/update operations."""

    def _ToJson(msg):
        return extra_types.JsonProtoEncoder(extra_types.JsonArray(entries=msg.entry))
    encoding.RegisterCustomMessageCodec(encoder=_ToJson, decoder=None)(cls.msgs.Write.ValuesValueListEntry)
    json_columns = table.GetJsonData(data)
    json_column_names = [col.col_name for col in json_columns]
    json_column_values = [col.col_value for col in json_columns]
    return cls.msgs.Write(columns=json_column_names, table=table.name, values=[cls.msgs.Write.ValuesValueListEntry(entry=json_column_values)])