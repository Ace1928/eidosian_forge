from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerStateWaiting(_messages.Message):
    """ContainerStateWaiting is a waiting state of a container.

  Fields:
    message: Message regarding why the container is not yet running.
    reason: Reason the container is not yet running.
  """
    message = _messages.StringField(1)
    reason = _messages.StringField(2)