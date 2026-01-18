from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsCreateRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsCreateRequest
  object.

  Fields:
    parent: Required. The environment name to create a Secret for, in the
      form: "projects/{projectId}/locations/{locationId}/environments/{environ
      mentId}"
    userWorkloadsSecret: A UserWorkloadsSecret resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    userWorkloadsSecret = _messages.MessageField('UserWorkloadsSecret', 2)