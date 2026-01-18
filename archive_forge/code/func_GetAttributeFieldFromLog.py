from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.privateca import text_utils
def GetAttributeFieldFromLog(field_name, is_log_entry, log_obj):
    return getattr(log_obj, GetProperField(field_name, is_log_entry), '')