from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsumerProject(_messages.Message):
    """Represents a consumer project.

  Fields:
    projectNum: Required. Project number of the consumer that is launching the
      service instance. It can own the network that is peered with Google or,
      be a service project in an XPN where the host project has the network.
  """
    projectNum = _messages.IntegerField(1)