from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventSource(_messages.Message):
    """Event Source referenceable within a WorkflowTrigger.

  Fields:
    eventSource: Output only. The fully qualified resource name for the event
      source.
    gitRepositoryLink: Output only. Resource name of Developer Connect
      GitRepositoryLink.
    id: identification to Resource.
    repository: Output only. Resource name of GCB v2 repo.
    subscription: Output only. Resource name of PubSub subscription.
  """
    eventSource = _messages.StringField(1)
    gitRepositoryLink = _messages.StringField(2)
    id = _messages.StringField(3)
    repository = _messages.StringField(4)
    subscription = _messages.StringField(5)