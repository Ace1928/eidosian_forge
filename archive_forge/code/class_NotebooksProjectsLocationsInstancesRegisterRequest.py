from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesRegisterRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesRegisterRequest object.

  Fields:
    parent: Required. Format:
      `parent=projects/{project_id}/locations/{location}`
    registerInstanceRequest: A RegisterInstanceRequest resource to be passed
      as the request body.
  """
    parent = _messages.StringField(1, required=True)
    registerInstanceRequest = _messages.MessageField('RegisterInstanceRequest', 2)