from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsKeyvaluemapsEntriesCreateRequest(_messages.Message):
    """A ApigeeOrganizationsKeyvaluemapsEntriesCreateRequest object.

  Fields:
    googleCloudApigeeV1KeyValueEntry: A GoogleCloudApigeeV1KeyValueEntry
      resource to be passed as the request body.
    parent: Required. Scope as indicated by the URI in which to create the key
      value map entry. Use **one** of the following structures in your
      request: *
      `organizations/{organization}/apis/{api}/keyvaluemaps/{keyvaluemap}`. *
      `organizations/{organization}/environments/{environment}/keyvaluemaps/{k
      eyvaluemap}` *
      `organizations/{organization}/keyvaluemaps/{keyvaluemap}`.
  """
    googleCloudApigeeV1KeyValueEntry = _messages.MessageField('GoogleCloudApigeeV1KeyValueEntry', 1)
    parent = _messages.StringField(2, required=True)