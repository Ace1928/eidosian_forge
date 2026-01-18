from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeFutureReservationsGetRequest(_messages.Message):
    """A ComputeFutureReservationsGetRequest object.

  Fields:
    futureReservation: Name of the future reservation to retrieve. Name should
      conform to RFC1035.
    project: Project ID for this request.
    zone: Name of the zone for this request. Name should conform to RFC1035.
  """
    futureReservation = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)