from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsLocationPrometheusApiV1SeriesRequest(_messages.Message):
    """A MonitoringProjectsLocationPrometheusApiV1SeriesRequest object.

  Fields:
    location: Location of the resource information. Has to be "global" for
      now.
    name: Required. The workspace on which to execute the request. It is not
      part of the open source API but used as a request path prefix to
      distinguish different virtual Prometheus instances of Google Prometheus
      Engine. The format is: projects/PROJECT_ID_OR_NUMBER.
    querySeriesRequest: A QuerySeriesRequest resource to be passed as the
      request body.
  """
    location = _messages.StringField(1, required=True)
    name = _messages.StringField(2, required=True)
    querySeriesRequest = _messages.MessageField('QuerySeriesRequest', 3)