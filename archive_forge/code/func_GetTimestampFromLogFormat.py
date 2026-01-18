from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.privateca import text_utils
def GetTimestampFromLogFormat(is_log_entry, log):
    """Returns timestamp in 'YYYY-MM-DD HH:MM:SS' string format."""
    timestamp = GetAttributeFieldFromLog('timestamp', is_log_entry, log)
    if is_log_entry:
        ts = timestamp_pb2.Timestamp()
        ts.FromJsonString(timestamp)
        log_entry_timestamp = ts.ToDatetime()
        return datetime.datetime.strftime(log_entry_timestamp, '%Y-%m-%d %H:%M:%S')
    return datetime.datetime.strftime(timestamp, '%Y-%m-%d %H:%M:%S')