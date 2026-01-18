from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeConfig(_messages.Message):
    """The configuration information for the Kubernetes Engine nodes running
  the Apache Airflow software.

  Fields:
    composerInternalIpv4CidrBlock: Optional. The IP range in CIDR notation to
      use internally by Cloud Composer. IP addresses are not reserved - and
      the same range can be used by multiple Cloud Composer environments. In
      case of overlap, IPs from this range will not be accessible in the
      user's VPC network. Cannot be updated. If not specified, the default
      value of '100.64.128.0/20' is used. This field is supported for Cloud
      Composer environments in versions composer-3.*.*-airflow-*.*.* and
      newer.
    composerNetworkAttachment: Optional. Network Attachment that Cloud
      Composer environment is connected to, which provides connectivity with a
      user's VPC network. Takes precedence over network and subnetwork
      settings. If not provided, but network and subnetwork are defined during
      environment, it will be provisioned. If not provided and network and
      subnetwork are also empty, then connectivity to user's VPC network is
      disabled. Network attachment must be provided in format projects/{projec
      t}/regions/{region}/networkAttachments/{networkAttachment}. This field
      is supported for Cloud Composer environments in versions
      composer-3.*.*-airflow-*.*.* and newer.
    diskSizeGb: Optional. The disk size in GB used for node VMs. Minimum size
      is 30GB. If unspecified, defaults to 100GB. Cannot be updated. This
      field is supported for Cloud Composer environments in versions
      composer-1.*.*-airflow-*.*.*.
    enableIpMasqAgent: Optional. Deploys 'ip-masq-agent' daemon set in the GKE
      cluster and defines nonMasqueradeCIDRs equals to pod IP range so IP
      masquerading is used for all destination addresses, except between pods
      traffic. See: https://cloud.google.com/kubernetes-engine/docs/how-to/ip-
      masquerade-agent
    ipAllocationPolicy: Optional. The IPAllocationPolicy fields for the GKE
      cluster.
    location: Optional. The Compute Engine [zone](/compute/docs/regions-zones)
      in which to deploy the VMs used to run the Apache Airflow software,
      specified as a [relative resource
      name](/apis/design/resource_names#relative_resource_name). For example:
      "projects/{projectId}/zones/{zoneId}". This `location` must belong to
      the enclosing environment's project and location. If both this field and
      `nodeConfig.machineType` are specified, `nodeConfig.machineType` must
      belong to this `location`; if both are unspecified, the service will
      pick a zone in the Compute Engine region corresponding to the Cloud
      Composer location, and propagate that choice to both fields. If only one
      field (`location` or `nodeConfig.machineType`) is specified, the
      location information from the specified field will be propagated to the
      unspecified field. This field is supported for Cloud Composer
      environments in versions composer-1.*.*-airflow-*.*.*.
    machineType: Optional. The Compute Engine [machine
      type](/compute/docs/machine-types) used for cluster instances, specified
      as a [relative resource
      name](/apis/design/resource_names#relative_resource_name). For example:
      "projects/{projectId}/zones/{zoneId}/machineTypes/{machineTypeId}". The
      `machineType` must belong to the enclosing environment's project and
      location. If both this field and `nodeConfig.location` are specified,
      this `machineType` must belong to the `nodeConfig.location`; if both are
      unspecified, the service will pick a zone in the Compute Engine region
      corresponding to the Cloud Composer location, and propagate that choice
      to both fields. If exactly one of this field and `nodeConfig.location`
      is specified, the location information from the specified field will be
      propagated to the unspecified field. The `machineTypeId` must not be a
      [shared-core machine type](/compute/docs/machine-types#sharedcore). If
      this field is unspecified, the `machineTypeId` defaults to
      "n1-standard-1". This field is supported for Cloud Composer environments
      in versions composer-1.*.*-airflow-*.*.*.
    maxPodsPerNode: Optional. The maximum number of pods per node in the Cloud
      Composer GKE cluster. The value must be between 8 and 110 and it can be
      set only if the environment is VPC-native. The default value is 32.
      Values of this field will be propagated both to the `default-pool` node
      pool of the newly created GKE cluster, and to the default "Maximum Pods
      per Node" value which is used for newly created node pools if their
      value is not explicitly set during node pool creation. For more
      information, see [Optimizing IP address allocation]
      (https://cloud.google.com/kubernetes-engine/docs/how-to/flexible-pod-
      cidr). Cannot be updated. This field is supported for Cloud Composer
      environments in versions composer-1.*.*-airflow-*.*.*.
    network: Optional. The Compute Engine network to be used for machine
      communications, specified as a [relative resource
      name](/apis/design/resource_names#relative_resource_name). For example:
      "projects/{projectId}/global/networks/{networkId}". If unspecified, the
      default network in the environment's project is used. If a [Custom
      Subnet Network](/vpc/docs/vpc#vpc_networks_and_subnets) is provided,
      `nodeConfig.subnetwork` must also be provided. For [Shared
      VPC](/vpc/docs/shared-vpc) subnetwork requirements, see
      `nodeConfig.subnetwork`.
    oauthScopes: Optional. The set of Google API scopes to be made available
      on all node VMs. If `oauth_scopes` is empty, defaults to
      ["https://www.googleapis.com/auth/cloud-platform"]. Cannot be updated.
      This field is supported for Cloud Composer environments in versions
      composer-1.*.*-airflow-*.*.*.
    serviceAccount: Optional. The Google Cloud Platform Service Account to be
      used by the workloads. If a service account is not specified, the
      "default" Compute Engine service account is used. Cannot be updated.
    subnetwork: Optional. The Compute Engine subnetwork to be used for machine
      communications, specified as a [relative resource
      name](/apis/design/resource_names#relative_resource_name). For example:
      "projects/{projectId}/regions/{regionId}/subnetworks/{subnetworkId}" If
      a subnetwork is provided, `nodeConfig.network` must also be provided,
      and the subnetwork must belong to the enclosing environment's project
      and location.
    tags: Optional. The list of instance tags applied to all node VMs. Tags
      are used to identify valid sources or targets for network firewalls.
      Each tag within the list must comply with
      [RFC1035](https://www.ietf.org/rfc/rfc1035.txt). Cannot be updated.
  """
    composerInternalIpv4CidrBlock = _messages.StringField(1)
    composerNetworkAttachment = _messages.StringField(2)
    diskSizeGb = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    enableIpMasqAgent = _messages.BooleanField(4)
    ipAllocationPolicy = _messages.MessageField('IPAllocationPolicy', 5)
    location = _messages.StringField(6)
    machineType = _messages.StringField(7)
    maxPodsPerNode = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    network = _messages.StringField(9)
    oauthScopes = _messages.StringField(10, repeated=True)
    serviceAccount = _messages.StringField(11)
    subnetwork = _messages.StringField(12)
    tags = _messages.StringField(13, repeated=True)