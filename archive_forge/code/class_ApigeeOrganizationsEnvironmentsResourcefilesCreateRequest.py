from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsResourcefilesCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsResourcefilesCreateRequest object.

  Fields:
    googleApiHttpBody: A GoogleApiHttpBody resource to be passed as the
      request body.
    name: Required. Name of the resource file. Must match the regular
      expression: [a-zA-Z0-9:/\\\\!@#$%^&{}\\[\\]()+\\-=,.~'` ]{1,255}
    parent: Required. Name of the environment in which to create the resource
      file in the following format: `organizations/{org}/environments/{env}`.
    type: Required. Resource file type. {{ resource_file_type }}
  """
    googleApiHttpBody = _messages.MessageField('GoogleApiHttpBody', 1)
    name = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    type = _messages.StringField(4)