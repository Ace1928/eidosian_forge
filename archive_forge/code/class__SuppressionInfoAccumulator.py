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
class _SuppressionInfoAccumulator(object):
    """Accumulates and outputs information about suppression for the tail session."""

    def __init__(self, get_now, output_warning, output_error):
        self._get_now = get_now
        self._warning = output_warning
        self._error = output_error
        self._count_by_reason_delta = collections.Counter()
        self._count_by_reason_cumulative = collections.Counter()
        self._last_flush = get_now()

    def _OutputSuppressionHelpMessage(self):
        self._warning('Find guidance for suppression at {}.'.format(_HELP_PAGE_LINK))

    def _ShouldFlush(self):
        return (self._get_now() - self._last_flush).total_seconds() > _SUPPRESSION_INFO_FLUSH_PERIOD_SECONDS

    def _OutputSuppressionDeltaMessage(self, reason_string, count):
        self._error('Suppressed {} entries due to {}.'.format(count, reason_string))

    def _OutputSuppressionCumulativeMessage(self, reason_string, count):
        self._warning('In total, suppressed {} messages due to {}.'.format(count, reason_string))

    def _Flush(self):
        self._last_flush = self._get_now()
        _HandleSuppressionCounts(self._count_by_reason_delta, self._OutputSuppressionDeltaMessage)
        self._count_by_reason_cumulative += self._count_by_reason_delta
        self._count_by_reason_delta.clear()

    def Finish(self):
        self._Flush()
        _HandleSuppressionCounts(self._count_by_reason_cumulative, self._OutputSuppressionCumulativeMessage)
        if self._count_by_reason_cumulative:
            self._OutputSuppressionHelpMessage()

    def Add(self, suppression_info):
        self._count_by_reason_delta += collections.Counter({info.reason: info.suppressed_count for info in suppression_info})
        if self._ShouldFlush():
            self._Flush()