from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchResultTypeValueValuesEnum(_messages.Enum):
    """Type of the search result. You can use this field to determine which
    get method to call to fetch the full resource.

    Values:
      SEARCH_RESULT_TYPE_UNSPECIFIED: Default unknown type.
      ENTRY: An Entry.
      TAG_TEMPLATE: A TagTemplate.
      ENTRY_GROUP: An EntryGroup.
    """
    SEARCH_RESULT_TYPE_UNSPECIFIED = 0
    ENTRY = 1
    TAG_TEMPLATE = 2
    ENTRY_GROUP = 3