from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Billing(_messages.Message):
    """Billing related configuration of the service.  The following example
  shows how to configure metrics for billing:      metrics:     - name:
  library.googleapis.com/read_calls       metric_kind: DELTA       value_type:
  INT64     - name: library.googleapis.com/write_calls       metric_kind:
  DELTA       value_type: INT64     billing:       metrics:       -
  library.googleapis.com/read_calls       - library.googleapis.com/write_calls
  The next example shows how to enable billing status check and customize the
  check behavior. It makes sure billing status check is included in the
  `Check` method of [Service Control API](https://cloud.google.com/service-
  control/). In the example, "google.storage.Get" method can be served when
  the billing status is either `current` or `delinquent`, while
  "google.storage.Write" method can only be served when the billing status is
  `current`:      billing:       rules:       - selector: google.storage.Get
  allowed_statuses:         - current         - delinquent       - selector:
  google.storage.Write         allowed_statuses: current  Mostly services
  should only allow `current` status when serving requests. In addition,
  services can choose to allow both `current` and `delinquent` statuses when
  serving read-only requests to resources. If there's no matching selector for
  operation, no billing status check will be performed.

  Fields:
    areaUnderCurveParams: Per resource grouping for delta billing based
      resource configs.
    metrics: Names of the metrics to report to billing. Each name must be
      defined in Service.metrics section.
    rules: A list of billing status rules for configuring billing status
      check.
  """
    areaUnderCurveParams = _messages.MessageField('AreaUnderCurveParams', 1, repeated=True)
    metrics = _messages.StringField(2, repeated=True)
    rules = _messages.MessageField('BillingStatusRule', 3, repeated=True)