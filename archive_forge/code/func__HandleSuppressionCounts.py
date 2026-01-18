from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import google.appengine.logging.v1.request_log_pb2
import google.cloud.appengine_v1.proto.audit_data_pb2
import google.cloud.appengine_v1alpha.proto.audit_data_pb2
import google.cloud.appengine_v1beta.proto.audit_data_pb2
import google.cloud.bigquery_logging_v1.proto.audit_data_pb2
import google.cloud.cloud_audit.proto.audit_log_pb2
import google.cloud.iam_admin_v1.proto.audit_data_pb2
import google.iam.v1.logging.audit_data_pb2
import google.type.money_pb2
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import gapic_util
from googlecloudsdk.core import log
import grpc
def _HandleSuppressionCounts(counts_by_reason, handler):
    """Handles supression counts."""
    client_class = apis.GetGapicClientClass('logging', 'v2')
    suppression_info = client_class.types.TailLogEntriesResponse.SuppressionInfo
    suppression_reason_strings = {suppression_info.Reason.RATE_LIMIT: 'Logging API backend rate limit', suppression_info.Reason.NOT_CONSUMED: 'client not consuming messages quickly enough'}
    for reason, count in counts_by_reason.items():
        reason_string = suppression_reason_strings.get(reason, 'UNKNOWN REASON: {}'.format(reason))
        handler(reason_string, count)