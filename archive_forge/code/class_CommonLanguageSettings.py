from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommonLanguageSettings(_messages.Message):
    """Required information for every language.

  Enums:
    DestinationsValueListEntryValuesEnum:

  Fields:
    destinations: The destination where API teams want this client library to
      be published.
    referenceDocsUri: Link to automatically generated reference documentation.
      Example: https://cloud.google.com/nodejs/docs/reference/asset/latest
  """

    class DestinationsValueListEntryValuesEnum(_messages.Enum):
        """DestinationsValueListEntryValuesEnum enum type.

    Values:
      CLIENT_LIBRARY_DESTINATION_UNSPECIFIED: Client libraries will neither be
        generated nor published to package managers.
      GITHUB: Generate the client library in a repo under
        github.com/googleapis, but don't publish it to package managers.
      PACKAGE_MANAGER: Publish the library to package managers like nuget.org
        and npmjs.com.
    """
        CLIENT_LIBRARY_DESTINATION_UNSPECIFIED = 0
        GITHUB = 1
        PACKAGE_MANAGER = 2
    destinations = _messages.EnumField('DestinationsValueListEntryValuesEnum', 1, repeated=True)
    referenceDocsUri = _messages.StringField(2)