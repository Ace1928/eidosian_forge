from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.privateca import text_utils
def FormatLog(log):
    """Format logs for a service."""
    is_log_entry = isinstance(log, apis.GetMessagesModule('logging', 'v2').LogEntry)
    log_entry_line = GetAttributeFieldFromLog('log_name', is_log_entry, log)
    if not log_entry_line:
        return ''
    split_log = log_entry_line.split('%2F')
    if len(split_log) < 2:
        return ''
    log_type = split_log[1]
    log_output = [GetTimestampFromLogFormat(is_log_entry, log)]
    if log_type == 'requests':
        http_request = GetAttributeFieldFromLog('http_request', is_log_entry, log)
        http_method = GetAttributeFieldFromLog('request_method', is_log_entry, http_request)
        status = GetAttributeFieldFromLog('status', is_log_entry, http_request)
        url = GetAttributeFieldFromLog('request_url', is_log_entry, http_request)
        log_output.append(http_method)
        log_output.append(str(status))
        log_output.append(url)
    elif log_type == 'stderr' or log_type == 'stdout':
        text_payload = GetAttributeFieldFromLog('text_payload', is_log_entry, log)
        log_output.append(text_payload)
    else:
        return ''
    return ' '.join(log_output)