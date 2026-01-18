from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Publishing(_messages.Message):
    """This message configures the settings for publishing [Google Cloud Client
  libraries](https://cloud.google.com/apis/docs/cloud-client-libraries)
  generated from the service config.

  Enums:
    OrganizationValueValuesEnum: For whom the client library is being
      published.

  Fields:
    apiShortName: Used as a tracking tag when collecting data about the APIs
      developer relations artifacts like docs, packages delivered to package
      managers, etc. Example: "speech".
    codeownerGithubTeams: GitHub teams to be added to CODEOWNERS in the
      directory in GitHub containing source code for the client libraries for
      this API.
    docTagPrefix: A prefix used in sample code when demarking regions to be
      included in documentation.
    documentationUri: Link to product home page. Example:
      https://cloud.google.com/asset-inventory/docs/overview
    githubLabel: GitHub label to apply to issues and pull requests opened for
      this API.
    librarySettings: Client library settings. If the same version string
      appears multiple times in this list, then the last one wins. Settings
      from earlier settings with the same version string are discarded.
    methodSettings: A list of API method settings, e.g. the behavior for
      methods that use the long-running operation pattern.
    newIssueUri: Link to a *public* URI where users can report issues.
      Example: https://issuetracker.google.com/issues/new?component=190865&tem
      plate=1161103
    organization: For whom the client library is being published.
    protoReferenceDocumentationUri: Optional link to proto reference
      documentation. Example:
      https://cloud.google.com/pubsub/lite/docs/reference/rpc
    restReferenceDocumentationUri: Optional link to REST reference
      documentation. Example:
      https://cloud.google.com/pubsub/lite/docs/reference/rest
  """

    class OrganizationValueValuesEnum(_messages.Enum):
        """For whom the client library is being published.

    Values:
      CLIENT_LIBRARY_ORGANIZATION_UNSPECIFIED: Not useful.
      CLOUD: Google Cloud Platform Org.
      ADS: Ads (Advertising) Org.
      PHOTOS: Photos Org.
      STREET_VIEW: Street View Org.
      SHOPPING: Shopping Org.
      GEO: Geo Org.
      GENERATIVE_AI: Generative AI - https://developers.generativeai.google
    """
        CLIENT_LIBRARY_ORGANIZATION_UNSPECIFIED = 0
        CLOUD = 1
        ADS = 2
        PHOTOS = 3
        STREET_VIEW = 4
        SHOPPING = 5
        GEO = 6
        GENERATIVE_AI = 7
    apiShortName = _messages.StringField(1)
    codeownerGithubTeams = _messages.StringField(2, repeated=True)
    docTagPrefix = _messages.StringField(3)
    documentationUri = _messages.StringField(4)
    githubLabel = _messages.StringField(5)
    librarySettings = _messages.MessageField('ClientLibrarySettings', 6, repeated=True)
    methodSettings = _messages.MessageField('MethodSettings', 7, repeated=True)
    newIssueUri = _messages.StringField(8)
    organization = _messages.EnumField('OrganizationValueValuesEnum', 9)
    protoReferenceDocumentationUri = _messages.StringField(10)
    restReferenceDocumentationUri = _messages.StringField(11)