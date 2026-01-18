from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouteSpec(_messages.Message):
    """RouteSpec holds the desired state of the Route (from the client).

  Fields:
    traffic: Traffic specifies how to distribute traffic over a collection of
      Knative Revisions and Configurations. Cloud Run currently supports a
      single configurationName.
  """
    traffic = _messages.MessageField('TrafficTarget', 1, repeated=True)