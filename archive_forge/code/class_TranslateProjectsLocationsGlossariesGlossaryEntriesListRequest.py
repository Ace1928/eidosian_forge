from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsGlossariesGlossaryEntriesListRequest(_messages.Message):
    """A TranslateProjectsLocationsGlossariesGlossaryEntriesListRequest object.

  Fields:
    pageSize: Optional. Requested page size. The server may return fewer
      glossary entries than requested. If unspecified, the server picks an
      appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return. Typically, this is the value of
      [ListGlossaryEntriesResponse.next_page_token] returned from the previous
      call. The first page is returned if `page_token`is empty or missing.
    parent: Required. The parent glossary resource name for listing the
      glossary's entries.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)