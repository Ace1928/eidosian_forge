from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformPublishersModelsListRequest(_messages.Message):
    """A AiplatformPublishersModelsListRequest object.

  Enums:
    ViewValueValuesEnum: Optional. PublisherModel view specifying which fields
      to read.

  Fields:
    filter: Optional. The standard list filter.
    languageCode: Optional. The IETF BCP-47 language code representing the
      language in which the publisher models' text information should be
      written in (see go/bcp47). If not set, by default English (en).
    orderBy: Optional. A comma-separated list of fields to order by, sorted in
      ascending order. Use "desc" after a field name for descending.
    pageSize: Optional. The standard list page size.
    pageToken: Optional. The standard list page token. Typically obtained via
      ListPublisherModelsResponse.next_page_token of the previous
      ModelGardenService.ListPublisherModels call.
    parent: Required. The name of the Publisher from which to list the
      PublisherModels. Format: `publishers/{publisher}`
    view: Optional. PublisherModel view specifying which fields to read.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. PublisherModel view specifying which fields to read.

    Values:
      PUBLISHER_MODEL_VIEW_UNSPECIFIED: The default / unset value. The API
        will default to the BASIC view.
      PUBLISHER_MODEL_VIEW_BASIC: Include basic metadata about the publisher
        model, but not the full contents.
      PUBLISHER_MODEL_VIEW_FULL: Include everything.
      PUBLISHER_MODEL_VERSION_VIEW_BASIC: Include: VersionId,
        ModelVersionExternalName, and SupportedActions.
    """
        PUBLISHER_MODEL_VIEW_UNSPECIFIED = 0
        PUBLISHER_MODEL_VIEW_BASIC = 1
        PUBLISHER_MODEL_VIEW_FULL = 2
        PUBLISHER_MODEL_VERSION_VIEW_BASIC = 3
    filter = _messages.StringField(1)
    languageCode = _messages.StringField(2)
    orderBy = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    parent = _messages.StringField(6, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 7)