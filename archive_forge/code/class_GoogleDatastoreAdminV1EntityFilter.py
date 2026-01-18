from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1EntityFilter(_messages.Message):
    """Identifies a subset of entities in a project. This is specified as
  combinations of kinds and namespaces (either or both of which may be all, as
  described in the following examples). Example usage: Entire project:
  kinds=[], namespace_ids=[] Kinds Foo and Bar in all namespaces:
  kinds=['Foo', 'Bar'], namespace_ids=[] Kinds Foo and Bar only in the default
  namespace: kinds=['Foo', 'Bar'], namespace_ids=[''] Kinds Foo and Bar in
  both the default and Baz namespaces: kinds=['Foo', 'Bar'],
  namespace_ids=['', 'Baz'] The entire Baz namespace: kinds=[],
  namespace_ids=['Baz']

  Fields:
    kinds: If empty, then this represents all kinds.
    namespaceIds: An empty list represents all namespaces. This is the
      preferred usage for projects that don't use namespaces. An empty string
      element represents the default namespace. This should be used if the
      project has data in non-default namespaces, but doesn't want to include
      them. Each namespace in this list must be unique.
  """
    kinds = _messages.StringField(1, repeated=True)
    namespaceIds = _messages.StringField(2, repeated=True)