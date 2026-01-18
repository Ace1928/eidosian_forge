from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSourcesCreateRequest(_messages.Message):
    """A SecuritycenterOrganizationsSourcesCreateRequest object.

  Fields:
    parent: Required. Resource name of the new source's parent. Its format
      should be "organizations/[organization_id]".
    source: A Source resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    source = _messages.MessageField('Source', 2)