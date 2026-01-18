from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Monitoring(_messages.Message):
    """Monitoring configuration of the service.  The example below shows how to
  configure monitored resources and metrics for monitoring. In the example, a
  monitored resource and two metrics are defined. The
  `library.googleapis.com/book/returned_count` metric is sent to both producer
  and consumer projects, whereas the
  `library.googleapis.com/book/overdue_count` metric is only sent to the
  consumer project.      monitored_resources:     - type:
  library.googleapis.com/branch       labels:       - key: /city
  description: The city where the library branch is located in.       - key:
  /name         description: The name of the branch.     metrics:     - name:
  library.googleapis.com/book/returned_count       metric_kind: DELTA
  value_type: INT64       labels:       - key: /customer_id     - name:
  library.googleapis.com/book/overdue_count       metric_kind: GAUGE
  value_type: INT64       labels:       - key: /customer_id     monitoring:
  producer_destinations:       - monitored_resource:
  library.googleapis.com/branch         metrics:         -
  library.googleapis.com/book/returned_count       consumer_destinations:
  - monitored_resource: library.googleapis.com/branch         metrics:
  - library.googleapis.com/book/returned_count         -
  library.googleapis.com/book/overdue_count

  Fields:
    consumerDestinations: Monitoring configurations for sending metrics to the
      consumer project. There can be multiple consumer destinations, each one
      must have a different monitored resource type. A metric can be used in
      at most one consumer destination.
    producerDestinations: Monitoring configurations for sending metrics to the
      producer project. There can be multiple producer destinations, each one
      must have a different monitored resource type. A metric can be used in
      at most one producer destination.
  """
    consumerDestinations = _messages.MessageField('MonitoringDestination', 1, repeated=True)
    producerDestinations = _messages.MessageField('MonitoringDestination', 2, repeated=True)