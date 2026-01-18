from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenomicsWorkersCheckInRequest(_messages.Message):
    """A GenomicsWorkersCheckInRequest object.

  Fields:
    checkInRequest: A CheckInRequest resource to be passed as the request
      body.
    id: The VM identity token for authenticating the VM instance.
      https://cloud.google.com/compute/docs/instances/verifying-instance-
      identity
  """
    checkInRequest = _messages.MessageField('CheckInRequest', 1)
    id = _messages.StringField(2, required=True)