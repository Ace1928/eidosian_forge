from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApiVersion(_messages.Message):
    """Describes a particular version of an API. ApiVersions are what consumers
  actually use.

  Messages:
    AnnotationsValue: Annotations attach non-identifying metadata to
      resources. Annotation keys and values are less restricted than those of
      labels, but should be generally used for small values of broad interest.
      Larger, topic- specific metadata should be stored in Artifacts.
    LabelsValue: Labels attach identifying metadata to resources. Identifying
      metadata can be used to filter list operations. Label keys and values
      can be no longer than 64 characters (Unicode codepoints), can only
      contain lowercase letters, numeric characters, underscores and dashes.
      International characters are allowed. No more than 64 user labels can be
      associated with one resource (System labels are excluded). See
      https://goo.gl/xmQnxf for more information and examples of labels.
      System reserved label keys are prefixed with
      `apigeeregistry.googleapis.com/` and cannot be changed.

  Fields:
    annotations: Annotations attach non-identifying metadata to resources.
      Annotation keys and values are less restricted than those of labels, but
      should be generally used for small values of broad interest. Larger,
      topic- specific metadata should be stored in Artifacts.
    createTime: Output only. Creation timestamp.
    description: A detailed description.
    displayName: Human-meaningful name.
    labels: Labels attach identifying metadata to resources. Identifying
      metadata can be used to filter list operations. Label keys and values
      can be no longer than 64 characters (Unicode codepoints), can only
      contain lowercase letters, numeric characters, underscores and dashes.
      International characters are allowed. No more than 64 user labels can be
      associated with one resource (System labels are excluded). See
      https://goo.gl/xmQnxf for more information and examples of labels.
      System reserved label keys are prefixed with
      `apigeeregistry.googleapis.com/` and cannot be changed.
    name: Resource name.
    primarySpec: The primary spec for this version. Format: projects/{project}
      /locations/{location}/apis/{api}/versions/{version}/specs/{spec}
    state: A user-definable description of the lifecycle phase of this API
      version. Format: free-form, but we expect single words that describe API
      maturity, e.g., "CONCEPT", "DESIGN", "DEVELOPMENT", "STAGING",
      "PRODUCTION", "DEPRECATED", "RETIRED".
    updateTime: Output only. Last update timestamp.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Annotations attach non-identifying metadata to resources. Annotation
    keys and values are less restricted than those of labels, but should be
    generally used for small values of broad interest. Larger, topic- specific
    metadata should be stored in Artifacts.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels attach identifying metadata to resources. Identifying metadata
    can be used to filter list operations. Label keys and values can be no
    longer than 64 characters (Unicode codepoints), can only contain lowercase
    letters, numeric characters, underscores and dashes. International
    characters are allowed. No more than 64 user labels can be associated with
    one resource (System labels are excluded). See https://goo.gl/xmQnxf for
    more information and examples of labels. System reserved label keys are
    prefixed with `apigeeregistry.googleapis.com/` and cannot be changed.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    primarySpec = _messages.StringField(7)
    state = _messages.StringField(8)
    updateTime = _messages.StringField(9)