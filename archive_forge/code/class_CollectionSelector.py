from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CollectionSelector(_messages.Message):
    """A selection of a collection, such as `messages as m1`.

  Fields:
    allDescendants: When false, selects only collections that are immediate
      children of the `parent` specified in the containing `RunQueryRequest`.
      When true, selects all descendant collections.
    collectionId: The collection ID. When set, selects only collections with
      this ID.
  """
    allDescendants = _messages.BooleanField(1)
    collectionId = _messages.StringField(2)