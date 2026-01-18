from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApiDeployment(_messages.Message):
    """Describes a service running at particular address that provides a
  particular version of an API. ApiDeployments have revisions which correspond
  to different configurations of a single deployment in time. Revision
  identifiers should be updated whenever the served API spec or endpoint
  address changes.

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
    accessGuidance: Text briefly describing how to access the endpoint.
      Changes to this value will not affect the revision.
    annotations: Annotations attach non-identifying metadata to resources.
      Annotation keys and values are less restricted than those of labels, but
      should be generally used for small values of broad interest. Larger,
      topic- specific metadata should be stored in Artifacts.
    apiSpecRevision: The full resource name (including revision ID) of the
      spec of the API being served by the deployment. Changes to this value
      will update the revision. Format: `projects/{project}/locations/{locatio
      n}/apis/{api}/versions/{version}/specs/{spec@revision}`
    createTime: Output only. Creation timestamp; when the deployment resource
      was created.
    description: A detailed description.
    displayName: Human-meaningful name.
    endpointUri: The address where the deployment is serving. Changes to this
      value will update the revision.
    externalChannelUri: The address of the external channel of the API (e.g.,
      the Developer Portal). Changes to this value will not affect the
      revision.
    intendedAudience: Text briefly identifying the intended audience of the
      API. Changes to this value will not affect the revision.
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
    revisionCreateTime: Output only. Revision creation timestamp; when the
      represented revision was created.
    revisionId: Output only. Immutable. The revision ID of the deployment. A
      new revision is committed whenever the deployment contents are changed.
      The format is an 8-character hexadecimal string.
    revisionUpdateTime: Output only. Last update timestamp: when the
      represented revision was last modified.
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
    accessGuidance = _messages.StringField(1)
    annotations = _messages.MessageField('AnnotationsValue', 2)
    apiSpecRevision = _messages.StringField(3)
    createTime = _messages.StringField(4)
    description = _messages.StringField(5)
    displayName = _messages.StringField(6)
    endpointUri = _messages.StringField(7)
    externalChannelUri = _messages.StringField(8)
    intendedAudience = _messages.StringField(9)
    labels = _messages.MessageField('LabelsValue', 10)
    name = _messages.StringField(11)
    revisionCreateTime = _messages.StringField(12)
    revisionId = _messages.StringField(13)
    revisionUpdateTime = _messages.StringField(14)