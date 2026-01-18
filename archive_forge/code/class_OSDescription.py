from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSDescription(_messages.Message):
    """A message describing the VM's OS. Including OS, Publisher, Offer and
  Plan if applicable.

  Fields:
    offer: OS offer.
    plan: OS plan.
    publisher: OS publisher.
    type: OS type.
  """
    offer = _messages.StringField(1)
    plan = _messages.StringField(2)
    publisher = _messages.StringField(3)
    type = _messages.StringField(4)