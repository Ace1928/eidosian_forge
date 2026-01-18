from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Reservation(_messages.Message):
    """Metadata about a reservation resource.

  Fields:
    name: The name of the reservation. Structured like: projects/{project_numb
      er}/locations/{location}/reservations/{reservation_id}
    throughputCapacity: The reserved throughput capacity. Every unit of
      throughput capacity is equivalent to 1 MiB/s of published messages or 2
      MiB/s of subscribed messages. Any topics which are declared as using
      capacity from a Reservation will consume resources from this reservation
      instead of being charged individually.
  """
    name = _messages.StringField(1)
    throughputCapacity = _messages.IntegerField(2)