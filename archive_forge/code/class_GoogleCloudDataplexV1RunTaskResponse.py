from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1RunTaskResponse(_messages.Message):
    """A GoogleCloudDataplexV1RunTaskResponse object.

  Fields:
    job: Jobs created by RunTask API.
  """
    job = _messages.MessageField('GoogleCloudDataplexV1Job', 1)