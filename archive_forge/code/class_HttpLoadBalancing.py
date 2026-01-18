from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpLoadBalancing(_messages.Message):
    """Configuration options for the HTTP (L7) load balancing controller addon,
  which makes it easy to set up HTTP load balancers for services in a cluster.

  Fields:
    disabled: Whether the HTTP Load Balancing controller is enabled in the
      cluster. When enabled, it runs a small pod in the cluster that manages
      the load balancers.
  """
    disabled = _messages.BooleanField(1)