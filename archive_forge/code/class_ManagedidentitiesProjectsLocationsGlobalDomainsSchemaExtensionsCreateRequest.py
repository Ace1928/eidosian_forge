from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsSchemaExtensionsCreateRequest(_messages.Message):
    """A
  ManagedidentitiesProjectsLocationsGlobalDomainsSchemaExtensionsCreateRequest
  object.

  Fields:
    parent: Required. The domain resource name using the form:
      `projects/{project_id}/locations/global/domains/{domain_name}`
    schemaExtension: A SchemaExtension resource to be passed as the request
      body.
    schemaExtensionId: Required. Unique id of the Schema Extension Request.
      This value should be 4-63 characters, and valid characters are
      /A-Z[0-9]-/.
  """
    parent = _messages.StringField(1, required=True)
    schemaExtension = _messages.MessageField('SchemaExtension', 2)
    schemaExtensionId = _messages.StringField(3)