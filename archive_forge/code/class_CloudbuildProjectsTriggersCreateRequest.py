from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsTriggersCreateRequest(_messages.Message):
    """A CloudbuildProjectsTriggersCreateRequest object.

  Fields:
    buildTrigger: A BuildTrigger resource to be passed as the request body.
    parent: The parent resource where this trigger will be created. Format:
      `projects/{project}/locations/{location}`
    projectId: Required. ID of the project for which to configure automatic
      builds.
  """
    buildTrigger = _messages.MessageField('BuildTrigger', 1)
    parent = _messages.StringField(2)
    projectId = _messages.StringField(3, required=True)