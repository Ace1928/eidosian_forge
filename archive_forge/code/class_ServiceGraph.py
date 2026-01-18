from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceGraph(_messages.Message):
    """Service Graph is one of the observability types, it stands for
  visualizing users service in a graph, in which services are the nodes and
  the edges are between nodes that communicate with each other.

  Fields:
    enabled: Defines whether the observability for service graph is enabled.
      If enabled, samples from user defined services will be collected and a
      graph visualization of those services will be built.
  """
    enabled = _messages.BooleanField(1)