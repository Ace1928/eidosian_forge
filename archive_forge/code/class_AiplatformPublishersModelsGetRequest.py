from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformPublishersModelsGetRequest(_messages.Message):
    """A AiplatformPublishersModelsGetRequest object.

  Enums:
    ViewValueValuesEnum: Optional. PublisherModel view specifying which fields
      to read.

  Fields:
    languageCode: Optional. The IETF BCP-47 language code representing the
      language in which the publisher model's text information should be
      written in (see go/bcp47).
    name: Required. The name of the PublisherModel resource. Format:
      `publishers/{publisher}/models/{publisher_model}`
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
    languageCode = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 3)