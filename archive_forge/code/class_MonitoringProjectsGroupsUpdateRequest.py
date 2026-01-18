from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsGroupsUpdateRequest(_messages.Message):
    """A MonitoringProjectsGroupsUpdateRequest object.

  Fields:
    group: A Group resource to be passed as the request body.
    name: Output only. The name of this group. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID] When creating a group,
      this field is ignored and a new name is created consisting of the
      project specified in the call to CreateGroup and a unique [GROUP_ID]
      that is generated automatically.
    validateOnly: If true, validate this request but do not update the
      existing group.
  """
    group = _messages.MessageField('Group', 1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)