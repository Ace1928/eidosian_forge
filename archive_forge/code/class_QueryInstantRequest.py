from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryInstantRequest(_messages.Message):
    """QueryInstantRequest holds all parameters of the Prometheus upstream
  instant query API plus GCM specific parameters.

  Fields:
    query: A PromQL query string. Query lanauge documentation:
      https://prometheus.io/docs/prometheus/latest/querying/basics/.
    time: The single point in time to evaluate the query for. Either floating
      point UNIX seconds or RFC3339 formatted timestamp.
    timeout: An upper bound timeout for the query. Either a Prometheus
      duration string
      (https://prometheus.io/docs/prometheus/latest/querying/basics/#time-
      durations) or floating point seconds. This non-standard encoding must be
      used for compatibility with the open source API. Clients may still
      implement timeouts at the connection level while ignoring this field.
  """
    query = _messages.StringField(1)
    time = _messages.StringField(2)
    timeout = _messages.StringField(3)