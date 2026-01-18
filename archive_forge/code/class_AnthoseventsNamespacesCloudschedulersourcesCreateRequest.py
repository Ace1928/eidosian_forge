from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudschedulersourcesCreateRequest(_messages.Message):
    """A AnthoseventsNamespacesCloudschedulersourcesCreateRequest object.

  Fields:
    cloudSchedulerSource: A CloudSchedulerSource resource to be passed as the
      request body.
    parent: The namespace in which this cloudschedulersource should be
      created.
  """
    cloudSchedulerSource = _messages.MessageField('CloudSchedulerSource', 1)
    parent = _messages.StringField(2, required=True)