from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BasicSli(_messages.Message):
    """An SLI measuring performance on a well-known service type. Performance
  will be computed on the basis of pre-defined metrics. The type of the
  service_resource determines the metrics to use and the
  service_resource.labels and metric_labels are used to construct a monitoring
  filter to filter that metric down to just the data relevant to this service.

  Fields:
    availability: Good service is defined to be the count of requests made to
      this service that return successfully.
    latency: Good service is defined to be the count of requests made to this
      service that are fast enough with respect to latency.threshold.
    location: OPTIONAL: The set of locations to which this SLI is relevant.
      Telemetry from other locations will not be used to calculate performance
      for this SLI. If omitted, this SLI applies to all locations in which the
      Service has activity. For service types that don't support breaking down
      by location, setting this field will result in an error.
    method: OPTIONAL: The set of RPCs to which this SLI is relevant. Telemetry
      from other methods will not be used to calculate performance for this
      SLI. If omitted, this SLI applies to all the Service's methods. For
      service types that don't support breaking down by method, setting this
      field will result in an error.
    version: OPTIONAL: The set of API versions to which this SLI is relevant.
      Telemetry from other API versions will not be used to calculate
      performance for this SLI. If omitted, this SLI applies to all API
      versions. For service types that don't support breaking down by version,
      setting this field will result in an error.
  """
    availability = _messages.MessageField('AvailabilityCriteria', 1)
    latency = _messages.MessageField('LatencyCriteria', 2)
    location = _messages.StringField(3, repeated=True)
    method = _messages.StringField(4, repeated=True)
    version = _messages.StringField(5, repeated=True)