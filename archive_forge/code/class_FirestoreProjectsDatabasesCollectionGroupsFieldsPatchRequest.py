from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesCollectionGroupsFieldsPatchRequest(_messages.Message):
    """A FirestoreProjectsDatabasesCollectionGroupsFieldsPatchRequest object.

  Fields:
    googleFirestoreAdminV1Field: A GoogleFirestoreAdminV1Field resource to be
      passed as the request body.
    name: Required. A field name of the form `projects/{project_id}/databases/
      {database_id}/collectionGroups/{collection_id}/fields/{field_path}` A
      field path may be a simple field name, e.g. `address` or a path to
      fields within map_value , e.g. `address.city`, or a special field path.
      The only valid special field is `*`, which represents any field. Field
      paths may be quoted using ` (backtick). The only character that needs to
      be escaped within a quoted field path is the backtick character itself,
      escaped using a backslash. Special characters in field paths that must
      be quoted include: `*`, `.`, ``` (backtick), `[`, `]`, as well as any
      ascii symbolic characters. Examples: (Note: Comments here are written in
      markdown syntax, so there is an additional layer of backticks to
      represent a code block) `\\`address.city\\`` represents a field named
      `address.city`, not the map key `city` in the field `address`. `\\`*\\``
      represents a field named `*`, not any field. A special `Field` contains
      the default indexing settings for all fields. This field's resource name
      is: `projects/{project_id}/databases/{database_id}/collectionGroups/__de
      fault__/fields/*` Indexes defined on this `Field` will be applied to all
      fields which do not have their own `Field` index configuration.
    updateMask: A mask, relative to the field. If specified, only
      configuration specified by this field_mask will be updated in the field.
  """
    googleFirestoreAdminV1Field = _messages.MessageField('GoogleFirestoreAdminV1Field', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)