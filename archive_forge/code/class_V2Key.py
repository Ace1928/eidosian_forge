from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2Key(_messages.Message):
    """The representation of a key managed by the API Keys API.

  Messages:
    AnnotationsValue: Annotations is an unstructured key-value map stored with
      a policy that may be set by external tools to store and retrieve
      arbitrary metadata. They are not queryable and should be preserved when
      modifying objects.

  Fields:
    annotations: Annotations is an unstructured key-value map stored with a
      policy that may be set by external tools to store and retrieve arbitrary
      metadata. They are not queryable and should be preserved when modifying
      objects.
    createTime: Output only. A timestamp identifying the time this key was
      originally created.
    deleteTime: Output only. A timestamp when this key was deleted. If the
      resource is not deleted, this must be empty.
    displayName: Human-readable display name of this key that you can modify.
      The maximum length is 63 characters.
    etag: Output only. A checksum computed by the server based on the current
      value of the Key resource. This may be sent on update and delete
      requests to ensure the client has an up-to-date value before proceeding.
      See https://google.aip.dev/154.
    keyString: Output only. An encrypted and signed value held by this key.
      This field can be accessed only through the `GetKeyString` method.
    name: Output only. The resource name of the key. The `name` has the form:
      `projects//locations/global/keys/`. For example: `projects/123456867718/
      locations/global/keys/b7ff1f9f-8275-410a-94dd-3855ee9b5dd2` NOTE: Key is
      a global resource; hence the only supported value for location is
      `global`.
    restrictions: Key restrictions.
    uid: Output only. Unique id in UUID4 format.
    updateTime: Output only. A timestamp identifying the time this key was
      last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Annotations is an unstructured key-value map stored with a policy that
    may be set by external tools to store and retrieve arbitrary metadata.
    They are not queryable and should be preserved when modifying objects.

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
    createTime = _messages.StringField(2)
    deleteTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    etag = _messages.StringField(5)
    keyString = _messages.StringField(6)
    name = _messages.StringField(7)
    restrictions = _messages.MessageField('V2Restrictions', 8)
    uid = _messages.StringField(9)
    updateTime = _messages.StringField(10)