from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GceClusterConfig(_messages.Message):
    """Common config settings for resources of Compute Engine cluster
  instances, applicable to all instances in the cluster.

  Enums:
    PrivateIpv6GoogleAccessValueValuesEnum: Optional. The type of IPv6 access
      for a cluster.

  Messages:
    MetadataValue: Optional. The Compute Engine metadata entries to add to all
      instances (see Project and instance metadata
      (https://cloud.google.com/compute/docs/storing-retrieving-
      metadata#project_and_instance_metadata)).

  Fields:
    confidentialInstanceConfig: Optional. Confidential Instance Config for
      clusters using Confidential VMs
      (https://cloud.google.com/compute/confidential-vm/docs).
    internalIpOnly: Optional. This setting applies to subnetwork-enabled
      networks. It is set to true by default in clusters created with image
      versions 2.2.x.When set to true: All cluster VMs have internal IP
      addresses. Google Private Access
      (https://cloud.google.com/vpc/docs/private-google-access) must be
      enabled to access Dataproc and other Google Cloud APIs. Off-cluster
      dependencies must be configured to be accessible without external IP
      addresses.When set to false: Cluster VMs are not restricted to internal
      IP addresses. Ephemeral external IP addresses are assigned to each
      cluster VM.
    metadata: Optional. The Compute Engine metadata entries to add to all
      instances (see Project and instance metadata
      (https://cloud.google.com/compute/docs/storing-retrieving-
      metadata#project_and_instance_metadata)).
    networkUri: Optional. The Compute Engine network to be used for machine
      communications. Cannot be specified with subnetwork_uri. If neither
      network_uri nor subnetwork_uri is specified, the "default" network of
      the project is used, if it exists. Cannot be a "Custom Subnet Network"
      (see Using Subnetworks
      (https://cloud.google.com/compute/docs/subnetworks) for more
      information).A full URL, partial URI, or short name are valid. Examples:
      https://www.googleapis.com/compute/v1/projects/[project_id]/global/netwo
      rks/default projects/[project_id]/global/networks/default default
    nodeGroupAffinity: Optional. Node Group Affinity for sole-tenant clusters.
    privateIpv6GoogleAccess: Optional. The type of IPv6 access for a cluster.
    reservationAffinity: Optional. Reservation Affinity for consuming Zonal
      reservation.
    serviceAccount: Optional. The Dataproc service account
      (https://cloud.google.com/dataproc/docs/concepts/configuring-
      clusters/service-accounts#service_accounts_in_dataproc) (also see VM
      Data Plane identity
      (https://cloud.google.com/dataproc/docs/concepts/iam/dataproc-
      principals#vm_service_account_data_plane_identity)) used by Dataproc
      cluster VM instances to access Google Cloud Platform services.If not
      specified, the Compute Engine default service account
      (https://cloud.google.com/compute/docs/access/service-
      accounts#default_service_account) is used.
    serviceAccountScopes: Optional. The URIs of service account scopes to be
      included in Compute Engine instances. The following base set of scopes
      is always included:
      https://www.googleapis.com/auth/cloud.useraccounts.readonly
      https://www.googleapis.com/auth/devstorage.read_write
      https://www.googleapis.com/auth/logging.writeIf no scopes are specified,
      the following defaults are also provided:
      https://www.googleapis.com/auth/bigquery
      https://www.googleapis.com/auth/bigtable.admin.table
      https://www.googleapis.com/auth/bigtable.data
      https://www.googleapis.com/auth/devstorage.full_control
    shieldedInstanceConfig: Optional. Shielded Instance Config for clusters
      using Compute Engine Shielded VMs
      (https://cloud.google.com/security/shielded-cloud/shielded-vm).
    subnetworkUri: Optional. The Compute Engine subnetwork to be used for
      machine communications. Cannot be specified with network_uri.A full URL,
      partial URI, or short name are valid. Examples: https://www.googleapis.c
      om/compute/v1/projects/[project_id]/regions/[region]/subnetworks/sub0
      projects/[project_id]/regions/[region]/subnetworks/sub0 sub0
    tags: The Compute Engine tags to add to all instances (see Tagging
      instances (https://cloud.google.com/compute/docs/label-or-tag-
      resources#tags)).
    zoneUri: Optional. The Compute Engine zone where the Dataproc cluster will
      be located. If omitted, the service will pick a zone in the cluster's
      Compute Engine region. On a get request, zone will always be present.A
      full URL, partial URI, or short name are valid. Examples:
      https://www.googleapis.com/compute/v1/projects/[project_id]/zones/[zone]
      projects/[project_id]/zones/[zone] [zone]
  """

    class PrivateIpv6GoogleAccessValueValuesEnum(_messages.Enum):
        """Optional. The type of IPv6 access for a cluster.

    Values:
      PRIVATE_IPV6_GOOGLE_ACCESS_UNSPECIFIED: If unspecified, Compute Engine
        default behavior will apply, which is the same as
        INHERIT_FROM_SUBNETWORK.
      INHERIT_FROM_SUBNETWORK: Private access to and from Google Services
        configuration inherited from the subnetwork configuration. This is the
        default Compute Engine behavior.
      OUTBOUND: Enables outbound private IPv6 access to Google Services from
        the Dataproc cluster.
      BIDIRECTIONAL: Enables bidirectional private IPv6 access between Google
        Services and the Dataproc cluster.
    """
        PRIVATE_IPV6_GOOGLE_ACCESS_UNSPECIFIED = 0
        INHERIT_FROM_SUBNETWORK = 1
        OUTBOUND = 2
        BIDIRECTIONAL = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Optional. The Compute Engine metadata entries to add to all instances
    (see Project and instance metadata
    (https://cloud.google.com/compute/docs/storing-retrieving-
    metadata#project_and_instance_metadata)).

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Additional properties of type MetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    confidentialInstanceConfig = _messages.MessageField('ConfidentialInstanceConfig', 1)
    internalIpOnly = _messages.BooleanField(2)
    metadata = _messages.MessageField('MetadataValue', 3)
    networkUri = _messages.StringField(4)
    nodeGroupAffinity = _messages.MessageField('NodeGroupAffinity', 5)
    privateIpv6GoogleAccess = _messages.EnumField('PrivateIpv6GoogleAccessValueValuesEnum', 6)
    reservationAffinity = _messages.MessageField('ReservationAffinity', 7)
    serviceAccount = _messages.StringField(8)
    serviceAccountScopes = _messages.StringField(9, repeated=True)
    shieldedInstanceConfig = _messages.MessageField('ShieldedInstanceConfig', 10)
    subnetworkUri = _messages.StringField(11)
    tags = _messages.StringField(12, repeated=True)
    zoneUri = _messages.StringField(13)