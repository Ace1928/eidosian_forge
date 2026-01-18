from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsLocationPrometheusApiV1LabelValuesRequest(_messages.Message):
    """A MonitoringProjectsLocationPrometheusApiV1LabelValuesRequest object.

  Fields:
    end: The end time to evaluate the query for. Either floating point UNIX
      seconds or RFC3339 formatted timestamp.
    label: The label name for which values are queried.
    location: Location of the resource information. Has to be "global" now.
    match: A list of matchers encoded in the Prometheus label matcher format
      to constrain the values to series that satisfy them.
    name: The workspace on which to execute the request. It is not part of the
      open source API but used as a request path prefix to distinguish
      different virtual Prometheus instances of Google Prometheus Engine. The
      format is: projects/PROJECT_ID_OR_NUMBER.
    start: The start time to evaluate the query for. Either floating point
      UNIX seconds or RFC3339 formatted timestamp.
  """
    end = _messages.StringField(1)
    label = _messages.StringField(2, required=True)
    location = _messages.StringField(3, required=True)
    match = _messages.StringField(4)
    name = _messages.StringField(5, required=True)
    start = _messages.StringField(6)