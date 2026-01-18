from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsGlossariesListRequest(_messages.Message):
    """A TranslateProjectsLocationsGlossariesListRequest object.

  Fields:
    filter: Optional. Filter specifying constraints of a list operation.
      Specify the constraint by the format of "key=value", where key must be
      "src" or "tgt", and the value must be a valid language code. For
      multiple restrictions, concatenate them by "AND" (uppercase only), such
      as: "src=en-US AND tgt=zh-CN". Notice that the exact match is used here,
      which means using 'en-US' and 'en' can lead to different results, which
      depends on the language code you used when you create the glossary. For
      the unidirectional glossaries, the "src" and "tgt" add restrictions on
      the source and target language code separately. For the equivalent term
      set glossaries, the "src" and/or "tgt" add restrictions on the term set.
      For example: "src=en-US AND tgt=zh-CN" will only pick the unidirectional
      glossaries which exactly match the source language code as "en-US" and
      the target language code "zh-CN", but all equivalent term set glossaries
      which contain "en-US" and "zh-CN" in their language set will be picked.
      If missing, no filtering is performed.
    pageSize: Optional. Requested page size. The server may return fewer
      glossaries than requested. If unspecified, the server picks an
      appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return. Typically, this is the value of
      [ListGlossariesResponse.next_page_token] returned from the previous call
      to `ListGlossaries` method. The first page is returned if `page_token`is
      empty or missing.
    parent: Required. The name of the project from which to list all of the
      glossaries.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)