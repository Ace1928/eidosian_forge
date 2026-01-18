from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApiSpec(_messages.Message):
    """Describes a version of an API in a structured way. ApiSpecs provide
  formal descriptions that consumers can use to use a version. ApiSpec
  resources are intended to be fully-resolved descriptions of an ApiVersion.
  When specs consist of multiple files, these should be bundled together
  (e.g., in a zip archive) and stored as a unit. Multiple specs can exist to
  provide representations in different API description formats.
  Synchronization of these representations would be provided by tooling and
  background services.

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
    contents: Input only. The contents of the spec. Provided by API callers
      when specs are created or updated. To access the contents of a spec, use
      GetApiSpecContents.
    createTime: Output only. Creation timestamp; when the spec resource was
      created.
    description: A detailed description.
    filename: A possibly-hierarchical name used to refer to the spec from
      other specs.
    hash: Output only. A SHA-256 hash of the spec's contents. If the spec is
      gzipped, this is the hash of the uncompressed spec.
    labels: Labels attach identifying metadata to resources. Identifying
      metadata can be used to filter list operations. Label keys and values
      can be no longer than 64 characters (Unicode codepoints), can only
      contain lowercase letters, numeric characters, underscores and dashes.
      International characters are allowed. No more than 64 user labels can be
      associated with one resource (System labels are excluded). See
      https://goo.gl/xmQnxf for more information and examples of labels.
      System reserved label keys are prefixed with
      `apigeeregistry.googleapis.com/` and cannot be changed.
    mimeType: A style (format) descriptor for this spec that is specified as a
      [Media Type](https://en.wikipedia.org/wiki/Media_type). Possible values
      include `application/vnd.apigee.proto`,
      `application/vnd.apigee.openapi`, and `application/vnd.apigee.graphql`,
      with possible suffixes representing compression types. These
      hypothetical names are defined in the vendor tree defined in RFC6838
      (https://tools.ietf.org/html/rfc6838) and are not final. Content types
      can specify compression. Currently only GZip compression is supported
      (indicated with "+gzip").
    name: Resource name.
    revisionCreateTime: Output only. Revision creation timestamp; when the
      represented revision was created.
    revisionId: Output only. Immutable. The revision ID of the spec. A new
      revision is committed whenever the spec contents are changed. The format
      is an 8-character hexadecimal string.
    revisionUpdateTime: Output only. Last update timestamp: when the
      represented revision was last modified.
    sizeBytes: Output only. The size of the spec file in bytes. If the spec is
      gzipped, this is the size of the uncompressed spec.
    sourceUri: The original source URI of the spec (if one exists). This is an
      external location that can be used for reference purposes but which may
      not be authoritative since this external resource may change after the
      spec is retrieved.
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
    contents = _messages.BytesField(2)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    filename = _messages.StringField(5)
    hash = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    mimeType = _messages.StringField(8)
    name = _messages.StringField(9)
    revisionCreateTime = _messages.StringField(10)
    revisionId = _messages.StringField(11)
    revisionUpdateTime = _messages.StringField(12)
    sizeBytes = _messages.IntegerField(13, variant=_messages.Variant.INT32)
    sourceUri = _messages.StringField(14)