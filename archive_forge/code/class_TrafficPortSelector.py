from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrafficPortSelector(_messages.Message):
    """Specification of a port-based selector.

  Fields:
    ports: Optional. A list of ports. Can be port numbers or port range
      (example, [80-90] specifies all ports from 80 to 90, including 80 and
      90) or named ports or * to specify all ports. If the list is empty, all
      ports are selected.
  """
    ports = _messages.StringField(1, repeated=True)