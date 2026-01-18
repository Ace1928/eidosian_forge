from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ResourceRuntime(_messages.Message):
    """Persistent Cluster runtime information as output

  Messages:
    AccessUrisValue: Output only. URIs for user to connect to the Cluster.
      Example: { "RAY_HEAD_NODE_INTERNAL_IP": "head-node-IP:10001"
      "RAY_DASHBOARD_URI": "ray-dashboard-address:8888" }

  Fields:
    accessUris: Output only. URIs for user to connect to the Cluster. Example:
      { "RAY_HEAD_NODE_INTERNAL_IP": "head-node-IP:10001" "RAY_DASHBOARD_URI":
      "ray-dashboard-address:8888" }
    notebookRuntimeTemplate: Output only. The resource name of
      NotebookRuntimeTemplate for the RoV Persistent Cluster The
      NotebokRuntimeTemplate is created in the same VPC (if set), and with the
      same Ray and Python version as the Persistent Cluster. Example:
      "projects/1000/locations/us-central1/notebookRuntimeTemplates/abc123"
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AccessUrisValue(_messages.Message):
        """Output only. URIs for user to connect to the Cluster. Example: {
    "RAY_HEAD_NODE_INTERNAL_IP": "head-node-IP:10001" "RAY_DASHBOARD_URI":
    "ray-dashboard-address:8888" }

    Messages:
      AdditionalProperty: An additional property for a AccessUrisValue object.

    Fields:
      additionalProperties: Additional properties of type AccessUrisValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AccessUrisValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    accessUris = _messages.MessageField('AccessUrisValue', 1)
    notebookRuntimeTemplate = _messages.StringField(2)