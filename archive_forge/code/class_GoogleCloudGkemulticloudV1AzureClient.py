from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureClient(_messages.Message):
    """`AzureClient` resources hold client authentication information needed by
  the Anthos Multi-Cloud API to manage Azure resources on your Azure
  subscription. When an AzureCluster is created, an `AzureClient` resource
  needs to be provided and all operations on Azure resources associated to
  that cluster will authenticate to Azure services using the given client.
  `AzureClient` resources are immutable and cannot be modified upon creation.
  Each `AzureClient` resource is bound to a single Azure Active Directory
  Application and tenant.

  Messages:
    AnnotationsValue: Optional. Annotations on the resource. This field has
      the same restrictions as Kubernetes annotations. The total size of all
      keys and values combined is limited to 256k. Keys can have 2 segments:
      prefix (optional) and name (required), separated by a slash (/). Prefix
      must be a DNS subdomain. Name must be 63 characters or less, begin and
      end with alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.

  Fields:
    annotations: Optional. Annotations on the resource. This field has the
      same restrictions as Kubernetes annotations. The total size of all keys
      and values combined is limited to 256k. Keys can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    applicationId: Required. The Azure Active Directory Application ID.
    createTime: Output only. The time at which this resource was created.
    name: The name of this resource. `AzureClient` resource names are
      formatted as `projects//locations//azureClients/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
    pemCertificate: Output only. The PEM encoded x509 certificate.
    reconciling: Output only. If set, there are currently pending changes to
      the client.
    tenantId: Required. The Azure Active Directory Tenant ID.
    uid: Output only. A globally unique identifier for the client.
    updateTime: Output only. The time at which this client was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. Annotations on the resource. This field has the same
    restrictions as Kubernetes annotations. The total size of all keys and
    values combined is limited to 256k. Keys can have 2 segments: prefix
    (optional) and name (required), separated by a slash (/). Prefix must be a
    DNS subdomain. Name must be 63 characters or less, begin and end with
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
    applicationId = _messages.StringField(2)
    createTime = _messages.StringField(3)
    name = _messages.StringField(4)
    pemCertificate = _messages.StringField(5)
    reconciling = _messages.BooleanField(6)
    tenantId = _messages.StringField(7)
    uid = _messages.StringField(8)
    updateTime = _messages.StringField(9)