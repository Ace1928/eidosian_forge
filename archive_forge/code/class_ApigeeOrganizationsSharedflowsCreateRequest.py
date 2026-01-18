from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSharedflowsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsSharedflowsCreateRequest object.

  Fields:
    action: Required. Must be set to either `import` or `validate`.
    googleApiHttpBody: A GoogleApiHttpBody resource to be passed as the
      request body.
    name: Required. The name to give the shared flow
    parent: Required. The name of the parent organization under which to
      create the shared flow. Must be of the form:
      `organizations/{organization_id}`
  """
    action = _messages.StringField(1)
    googleApiHttpBody = _messages.MessageField('GoogleApiHttpBody', 2)
    name = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)