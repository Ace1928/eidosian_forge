from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsCreateRequest(_messages.Message):
    """A
  ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsCreateRequest
  object.

  Fields:
    parent: Required. The environment name to create a ConfigMap for, in the
      form: "projects/{projectId}/locations/{locationId}/environments/{environ
      mentId}"
    userWorkloadsConfigMap: A UserWorkloadsConfigMap resource to be passed as
      the request body.
  """
    parent = _messages.StringField(1, required=True)
    userWorkloadsConfigMap = _messages.MessageField('UserWorkloadsConfigMap', 2)