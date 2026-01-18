from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserWorkloadsSecret(_messages.Message):
    """User workloads Secret used by Airflow tasks that run with Kubernetes
  executor or KubernetesPodOperator.

  Messages:
    DataValue: Optional. The "data" field of Kubernetes Secret, organized in
      key-value pairs, which can contain sensitive values such as a password,
      a token, or a key. The values for all keys have to be base64-encoded
      strings. For details see:
      https://kubernetes.io/docs/concepts/configuration/secret/

  Fields:
    data: Optional. The "data" field of Kubernetes Secret, organized in key-
      value pairs, which can contain sensitive values such as a password, a
      token, or a key. The values for all keys have to be base64-encoded
      strings. For details see:
      https://kubernetes.io/docs/concepts/configuration/secret/
    name: Identifier. The resource name of the Secret, in the form: "projects/
      {projectId}/locations/{locationId}/environments/{environmentId}/userWork
      loadsSecrets/{userWorkloadsSecretId}"
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DataValue(_messages.Message):
        """Optional. The "data" field of Kubernetes Secret, organized in key-
    value pairs, which can contain sensitive values such as a password, a
    token, or a key. The values for all keys have to be base64-encoded
    strings. For details see:
    https://kubernetes.io/docs/concepts/configuration/secret/

    Messages:
      AdditionalProperty: An additional property for a DataValue object.

    Fields:
      additionalProperties: Additional properties of type DataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    data = _messages.MessageField('DataValue', 1)
    name = _messages.StringField(2)