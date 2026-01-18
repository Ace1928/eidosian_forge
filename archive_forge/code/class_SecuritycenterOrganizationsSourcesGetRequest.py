from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSourcesGetRequest(_messages.Message):
    """A SecuritycenterOrganizationsSourcesGetRequest object.

  Fields:
    name: Required. Relative resource name of the source. Its format is
      "organizations/[organization_id]/source/[source_id]".
  """
    name = _messages.StringField(1, required=True)