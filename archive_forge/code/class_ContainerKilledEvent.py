from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerKilledEvent(_messages.Message):
    """An event generated when a container is forcibly terminated by the
  worker. Currently, this only occurs when the container outlives the timeout
  specified by the user.

  Fields:
    actionId: The numeric ID of the action that started the container.
  """
    actionId = _messages.IntegerField(1, variant=_messages.Variant.INT32)