from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListSharedFlowsResponse(_messages.Message):
    """To change this message, in the same CL add a change log in go/changing-
  api-proto-breaks-ui

  Fields:
    sharedFlows: A GoogleCloudApigeeV1SharedFlow attribute.
  """
    sharedFlows = _messages.MessageField('GoogleCloudApigeeV1SharedFlow', 1, repeated=True)