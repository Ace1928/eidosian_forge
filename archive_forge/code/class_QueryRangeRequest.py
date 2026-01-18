from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryRangeRequest(_messages.Message):
    """QueryRangeRequest holds all parameters of the Prometheus upstream range
  query API plus GCM specific parameters.

  Fields:
    end: The end time to evaluate the query for. Either floating point UNIX
      seconds or RFC3339 formatted timestamp.
    query: A PromQL query string. Query lanauge documentation:
      https://prometheus.io/docs/prometheus/latest/querying/basics/.
    start: The start time to evaluate the query for. Either floating point
      UNIX seconds or RFC3339 formatted timestamp.
    step: The resolution of query result. Either a Prometheus duration string
      (https://prometheus.io/docs/prometheus/latest/querying/basics/#time-
      durations) or floating point seconds. This non-standard encoding must be
      used for compatibility with the open source API. Clients may still
      implement timeouts at the connection level while ignoring this field.
    timeout: An upper bound timeout for the query. Either a Prometheus
      duration string
      (https://prometheus.io/docs/prometheus/latest/querying/basics/#time-
      durations) or floating point seconds. This non-standard encoding must be
      used for compatibility with the open source API. Clients may still
      implement timeouts at the connection level while ignoring this field.
  """
    end = _messages.StringField(1)
    query = _messages.StringField(2)
    start = _messages.StringField(3)
    step = _messages.StringField(4)
    timeout = _messages.StringField(5)