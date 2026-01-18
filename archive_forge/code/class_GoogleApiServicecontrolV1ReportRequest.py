from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1ReportRequest(_messages.Message):
    """Request message for the Report method.

  Fields:
    operations: Operations to be reported. Typically the service should report
      one operation per request. Putting multiple operations into a single
      request is allowed, but should be used only when multiple operations are
      natually available at the time of the report. There is no limit on the
      number of operations in the same ReportRequest, however the
      ReportRequest size should be no larger than 1MB. See
      ReportResponse.report_errors for partial failure behavior.
    serviceConfigId: Specifies which version of service config should be used
      to process the request. If unspecified or no matching version can be
      found, the latest one will be used.
    serviceName: The service name as specified in its service configuration.
      For example, `"pubsub.googleapis.com"`. See
      [google.api.Service](https://cloud.google.com/service-
      management/reference/rpc/google.api#google.api.Service) for the
      definition of a service name.
  """
    operations = _messages.MessageField('GoogleApiServicecontrolV1Operation', 1, repeated=True)
    serviceConfigId = _messages.StringField(2)
    serviceName = _messages.StringField(3)