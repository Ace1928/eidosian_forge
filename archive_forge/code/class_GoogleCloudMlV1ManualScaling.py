from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ManualScaling(_messages.Message):
    """Options for manually scaling a model.

  Fields:
    nodes: The number of nodes to allocate for this model. These nodes are
      always up, starting from the time the model is deployed, so the cost of
      operating this model will be proportional to `nodes` * number of hours
      since last billing cycle plus the cost for each prediction performed.
  """
    nodes = _messages.IntegerField(1, variant=_messages.Variant.INT32)