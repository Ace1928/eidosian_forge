from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CustomResourceDefinitionNames(_messages.Message):
    """CustomResourceDefinitionNames indicates the names to serve this
  CustomResourceDefinition

  Fields:
    categories: Categories is a list of grouped resources custom resources
      belong to (e.g. 'all') +optional
    kind: Kind is the serialized kind of the resource. It is normally
      CamelCase and singular.
    listKind: ListKind is the serialized kind of the list for this resource.
      Defaults to List. +optional
    plural: Plural is the plural name of the resource to serve. It must match
      the name of the CustomResourceDefinition-registration too: plural.group
      and it must be all lowercase.
    shortNames: ShortNames are short names for the resource. It must be all
      lowercase. +optional
    singular: Singular is the singular name of the resource. It must be all
      lowercase Defaults to lowercased +optional
  """
    categories = _messages.StringField(1, repeated=True)
    kind = _messages.StringField(2)
    listKind = _messages.StringField(3)
    plural = _messages.StringField(4)
    shortNames = _messages.StringField(5, repeated=True)
    singular = _messages.StringField(6)