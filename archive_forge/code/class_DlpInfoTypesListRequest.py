from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpInfoTypesListRequest(_messages.Message):
    """A DlpInfoTypesListRequest object.

  Fields:
    filter: filter to only return infoTypes supported by certain parts of the
      API. Defaults to supported_by=INSPECT.
    languageCode: BCP-47 language code for localized infoType friendly names.
      If omitted, or if localized strings are not available, en-US strings
      will be returned.
    locationId: Deprecated. This field has no effect.
    parent: The parent resource name. The format of this value is as follows:
      locations/ LOCATION_ID
  """
    filter = _messages.StringField(1)
    languageCode = _messages.StringField(2)
    locationId = _messages.StringField(3)
    parent = _messages.StringField(4)