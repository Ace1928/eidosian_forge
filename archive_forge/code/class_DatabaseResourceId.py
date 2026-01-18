from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatabaseResourceId(_messages.Message):
    """DatabaseResourceId will serve as primary key for any resource ingestion
  event.

  Enums:
    ProviderValueValuesEnum: Required. Cloud provider name. Ex:
      GCP/AWS/Azure/OnPrem/SelfManaged

  Fields:
    provider: Required. Cloud provider name. Ex:
      GCP/AWS/Azure/OnPrem/SelfManaged
    providerDescription: Optional. Needs to be used only when the provider is
      PROVIDER_OTHER.
    resourceType: Required. The type of resource this ID is identifying. Ex
      redis.googleapis.com/Instance, redis.googleapis.com/Cluster,
      alloydb.googleapis.com/Cluster, alloydb.googleapis.com/Instance,
      spanner.googleapis.com/Instance REQUIRED Please refer go/condor-common-
      datamodel
    uniqueId: Required. A service-local token that distinguishes this resource
      from other resources within the same service.
  """

    class ProviderValueValuesEnum(_messages.Enum):
        """Required. Cloud provider name. Ex: GCP/AWS/Azure/OnPrem/SelfManaged

    Values:
      PROVIDER_UNSPECIFIED: <no description>
      GCP: Google cloud platform provider
      AWS: Amazon web service
      AZURE: Azure web service
      ONPREM: On-prem database resources.
      SELFMANAGED: Self-managed database provider. These are resources on a
        cloud platform, e.g., database resource installed in a GCE VM, but not
        a managed database service.
      PROVIDER_OTHER: For the rest of the other categories. Other refers to
        the rest of other database service providers, this could be smaller
        cloud provider. This needs to be provided when the provider is known,
        but it is not present in the existing set of enum values.
    """
        PROVIDER_UNSPECIFIED = 0
        GCP = 1
        AWS = 2
        AZURE = 3
        ONPREM = 4
        SELFMANAGED = 5
        PROVIDER_OTHER = 6
    provider = _messages.EnumField('ProviderValueValuesEnum', 1)
    providerDescription = _messages.StringField(2)
    resourceType = _messages.StringField(3)
    uniqueId = _messages.StringField(4)