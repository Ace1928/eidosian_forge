from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsGroupsCreateRequest(_messages.Message):
    """A MonitoringProjectsGroupsCreateRequest object.

  Fields:
    group: A Group resource to be passed as the request body.
    name: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) in which to
      create the group. The format is: projects/[PROJECT_ID_OR_NUMBER]
    validateOnly: If true, validate this request but do not create the group.
  """
    group = _messages.MessageField('Group', 1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)