from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsKeyvaluemapsEntriesGetRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsKeyvaluemapsEntriesGetRequest object.

  Fields:
    name: Required. Scope as indicated by the URI in which to fetch the key
      value map entry/value. Use **one** of the following structures in your
      request: * `organizations/{organization}/apis/{api}/keyvaluemaps/{keyval
      uemap}/entries/{entry}`. * `organizations/{organization}/environments/{e
      nvironment}/keyvaluemaps/{keyvaluemap}/entries/{entry}` * `organizations
      /{organization}/keyvaluemaps/{keyvaluemap}/entries/{entry}`.
  """
    name = _messages.StringField(1, required=True)