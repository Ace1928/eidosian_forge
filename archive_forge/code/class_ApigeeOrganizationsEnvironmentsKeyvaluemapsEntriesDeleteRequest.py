from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsKeyvaluemapsEntriesDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsKeyvaluemapsEntriesDeleteRequest
  object.

  Fields:
    name: Required. Scope as indicated by the URI in which to delete the key
      value map entry. Use **one** of the following structures in your
      request: * `organizations/{organization}/apis/{api}/keyvaluemaps/{keyval
      uemap}/entries/{entry}`. * `organizations/{organization}/environments/{e
      nvironment}/keyvaluemaps/{keyvaluemap}/entries/{entry}` * `organizations
      /{organization}/keyvaluemaps/{keyvaluemap}/entries/{entry}`.
  """
    name = _messages.StringField(1, required=True)