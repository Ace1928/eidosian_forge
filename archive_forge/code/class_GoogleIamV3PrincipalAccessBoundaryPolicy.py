from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3PrincipalAccessBoundaryPolicy(_messages.Message):
    """An IAM principal access boundary policy resource.

  Messages:
    AnnotationsValue: Optional. User defined annotations. See
      https://google.aip.dev/148#annotations for more details such as format
      and size limitations

  Fields:
    annotations: Optional. User defined annotations. See
      https://google.aip.dev/148#annotations for more details such as format
      and size limitations
    createTime: Output only. The time when the principal access boundary
      policy was created.
    details: Optional. The details for the principal access boundary policy.
    displayName: Optional. The description of the principal access boundary
      policy. Must be less than or equal to 63 characters.
    etag: Optional. The etag for the principal access boundary. If this is
      provided on update, it must match the server's etag.
    name: Identifier. The resource name of the principal access boundary
      policy. The following format is supported: `organizations/{organization_
      id}/locations/{location}/principalAccessBoundaryPolicies/{policy_id}`
    uid: Output only. The globally unique ID of the principal access boundary
      policy.
    updateTime: Output only. The time when the principal access boundary
      policy was most recently updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. User defined annotations. See
    https://google.aip.dev/148#annotations for more details such as format and
    size limitations

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
    details = _messages.MessageField('GoogleIamV3PrincipalAccessBoundaryPolicyDetails', 3)
    displayName = _messages.StringField(4)
    etag = _messages.StringField(5)
    name = _messages.StringField(6)
    uid = _messages.StringField(7)
    updateTime = _messages.StringField(8)