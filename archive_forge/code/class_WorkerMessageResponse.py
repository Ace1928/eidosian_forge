from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkerMessageResponse(_messages.Message):
    """A worker_message response allows the server to pass information to the
  sender.

  Fields:
    streamingScalingReportResponse: Service's streaming scaling response for
      workers.
    workerHealthReportResponse: The service's response to a worker's health
      report.
    workerMetricsResponse: Service's response to reporting worker metrics
      (currently empty).
    workerShutdownNoticeResponse: Service's response to shutdown notice
      (currently empty).
    workerThreadScalingReportResponse: Service's thread scaling recommendation
      for workers.
  """
    streamingScalingReportResponse = _messages.MessageField('StreamingScalingReportResponse', 1)
    workerHealthReportResponse = _messages.MessageField('WorkerHealthReportResponse', 2)
    workerMetricsResponse = _messages.MessageField('ResourceUtilizationReportResponse', 3)
    workerShutdownNoticeResponse = _messages.MessageField('WorkerShutdownNoticeResponse', 4)
    workerThreadScalingReportResponse = _messages.MessageField('WorkerThreadScalingReportResponse', 5)