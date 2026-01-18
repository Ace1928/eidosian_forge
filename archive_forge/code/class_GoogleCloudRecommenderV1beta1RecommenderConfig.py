from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1beta1RecommenderConfig(_messages.Message):
    """Configuration for a Recommender.

  Messages:
    AnnotationsValue: Allows clients to store small amounts of arbitrary data.
      Annotations must follow the Kubernetes syntax. The total size of all
      keys and values combined is limited to 256k. Key can have 2 segments:
      prefix (optional) and name (required), separated by a slash (/). Prefix
      must be a DNS subdomain. Name must be 63 characters or less, begin and
      end with alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.

  Fields:
    annotations: Allows clients to store small amounts of arbitrary data.
      Annotations must follow the Kubernetes syntax. The total size of all
      keys and values combined is limited to 256k. Key can have 2 segments:
      prefix (optional) and name (required), separated by a slash (/). Prefix
      must be a DNS subdomain. Name must be 63 characters or less, begin and
      end with alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    displayName: A user-settable field to provide a human-readable name to be
      used in user interfaces.
    etag: Fingerprint of the RecommenderConfig. Provides optimistic locking
      when updating.
    name: Name of recommender config. Eg, projects/[PROJECT_NUMBER]/locations/
      [LOCATION]/recommenders/[RECOMMENDER_ID]/config
    recommenderGenerationConfig: RecommenderGenerationConfig which configures
      the Generation of recommendations for this recommender.
    revisionId: Output only. Immutable. The revision ID of the config. A new
      revision is committed whenever the config is changed in any way. The
      format is an 8-character hexadecimal string.
    updateTime: Last time when the config was updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Allows clients to store small amounts of arbitrary data. Annotations
    must follow the Kubernetes syntax. The total size of all keys and values
    combined is limited to 256k. Key can have 2 segments: prefix (optional)
    and name (required), separated by a slash (/). Prefix must be a DNS
    subdomain. Name must be 63 characters or less, begin and end with
    alphanumerics, with dashes (-), underscores (_), dots (.), and
    alphanumerics between.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    displayName = _messages.StringField(2)
    etag = _messages.StringField(3)
    name = _messages.StringField(4)
    recommenderGenerationConfig = _messages.MessageField('GoogleCloudRecommenderV1beta1RecommenderGenerationConfig', 5)
    revisionId = _messages.StringField(6)
    updateTime = _messages.StringField(7)