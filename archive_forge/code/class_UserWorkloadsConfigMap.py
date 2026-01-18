from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserWorkloadsConfigMap(_messages.Message):
    """User workloads ConfigMap used by Airflow tasks that run with Kubernetes
  executor or KubernetesPodOperator.

  Messages:
    DataValue: Optional. The "data" field of Kubernetes ConfigMap, organized
      in key-value pairs. For details see:
      https://kubernetes.io/docs/concepts/configuration/configmap/

  Fields:
    data: Optional. The "data" field of Kubernetes ConfigMap, organized in
      key-value pairs. For details see:
      https://kubernetes.io/docs/concepts/configuration/configmap/
    name: Identifier. The resource name of the ConfigMap, in the form: "projec
      ts/{projectId}/locations/{locationId}/environments/{environmentId}/userW
      orkloadsConfigMaps/{userWorkloadsConfigMapId}"
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DataValue(_messages.Message):
        """Optional. The "data" field of Kubernetes ConfigMap, organized in key-
    value pairs. For details see:
    https://kubernetes.io/docs/concepts/configuration/configmap/

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