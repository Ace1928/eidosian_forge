from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApisCreateRequest(_messages.Message):
    """A ApigeeOrganizationsApisCreateRequest object.

  Fields:
    action: Action to perform when importing an API proxy configuration
      bundle. Set this parameter to one of the following values: * `import` to
      import the API proxy configuration bundle. * `validate` to validate the
      API proxy configuration bundle without importing it.
    googleApiHttpBody: A GoogleApiHttpBody resource to be passed as the
      request body.
    name: Name of the API proxy. Restrict the characters used to: A-Za-z0-9._-
    parent: Required. Name of the organization in the following format:
      `organizations/{org}`
    validate: Ignored. All uploads are validated regardless of the value of
      this field. Maintained for compatibility with Apigee Edge API.
  """
    action = _messages.StringField(1)
    googleApiHttpBody = _messages.MessageField('GoogleApiHttpBody', 2)
    name = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    validate = _messages.BooleanField(5)