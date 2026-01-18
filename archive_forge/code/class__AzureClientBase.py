from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.azure import resource_args
from googlecloudsdk.command_lib.container.gkemulticloud import flags
class _AzureClientBase(client.ClientBase):
    """Base class for Azure gkemulticloud API clients."""

    def _Cluster(self, cluster_ref, args):
        azure_client = resource_args.ParseAzureClientResourceArg(args).RelativeName() if hasattr(args, 'client') and args.IsSpecified('client') else None
        cluster_type = self._messages.GoogleCloudGkemulticloudV1AzureCluster
        kwargs = {'annotations': self._Annotations(args, cluster_type), 'authorization': self._Authorization(args), 'azureClient': azure_client, 'azureServicesAuthentication': self._AzureServicesAuthentication(args), 'azureRegion': flags.GetAzureRegion(args), 'controlPlane': self._ControlPlane(args), 'description': flags.GetDescription(args), 'fleet': self._Fleet(args), 'loggingConfig': flags.GetLogging(args), 'monitoringConfig': flags.GetMonitoringConfig(args), 'name': cluster_ref.azureClustersId, 'networking': self._ClusterNetworking(args), 'resourceGroupId': flags.GetResourceGroupId(args)}
        return self._messages.GoogleCloudGkemulticloudV1AzureCluster(**kwargs) if any(kwargs.values()) else None

    def _AzureServicesAuthentication(self, args):
        kwargs = {'applicationId': flags.GetAzureApplicationID(args), 'tenantId': flags.GetAzureTenantID(args)}
        if not any(kwargs.values()):
            return None
        return self._messages.GoogleCloudGkemulticloudV1AzureServicesAuthentication(**kwargs)

    def _Client(self, client_ref, args):
        kwargs = {'applicationId': getattr(args, 'app_id', None), 'name': client_ref.azureClientsId, 'tenantId': getattr(args, 'tenant_id', None)}
        return self._messages.GoogleCloudGkemulticloudV1AzureClient(**kwargs) if any(kwargs.values()) else None

    def _NodePool(self, node_pool_ref, args):
        nodepool_type = self._messages.GoogleCloudGkemulticloudV1AzureNodePool
        kwargs = {'annotations': self._Annotations(args, nodepool_type), 'autoscaling': self._Autoscaling(args), 'azureAvailabilityZone': flags.GetAzureAvailabilityZone(args), 'config': self._NodeConfig(args), 'management': self._NodeManagement(args), 'maxPodsConstraint': self._MaxPodsConstraint(args), 'name': node_pool_ref.azureNodePoolsId, 'subnetId': flags.GetSubnetID(args), 'version': flags.GetNodeVersion(args)}
        return self._messages.GoogleCloudGkemulticloudV1AzureNodePool(**kwargs) if any(kwargs.values()) else None

    def _DiskTemplate(self, args, kind):
        kwargs = {}
        if kind == 'root':
            kwargs['sizeGib'] = flags.GetRootVolumeSize(args)
        elif kind == 'main':
            kwargs['sizeGib'] = flags.GetMainVolumeSize(args)
        return self._messages.GoogleCloudGkemulticloudV1AzureDiskTemplate(**kwargs) if any(kwargs.values()) else None

    def _ProxyConfig(self, args):
        kwargs = {'resourceGroupId': flags.GetProxyResourceGroupId(args), 'secretId': flags.GetProxySecretId(args)}
        return self._messages.GoogleCloudGkemulticloudV1AzureProxyConfig(**kwargs) if any(kwargs.values()) else None

    def _ConfigEncryption(self, args):
        kwargs = {'keyId': flags.GetConfigEncryptionKeyId(args), 'publicKey': flags.GetConfigEncryptionPublicKey(args)}
        return self._messages.GoogleCloudGkemulticloudV1AzureConfigEncryption(**kwargs) if any(kwargs.values()) else None

    def _Authorization(self, args):
        admin_users = flags.GetAdminUsers(args)
        admin_groups = flags.GetAdminGroups(args)
        if not admin_users and (not admin_groups):
            return None
        kwargs = {}
        if admin_users:
            kwargs['adminUsers'] = [self._messages.GoogleCloudGkemulticloudV1AzureClusterUser(username=u) for u in admin_users]
        if admin_groups:
            kwargs['adminGroups'] = [self._messages.GoogleCloudGkemulticloudV1AzureClusterGroup(group=g) for g in admin_groups]
        if not any(kwargs.values()):
            return None
        return self._messages.GoogleCloudGkemulticloudV1AzureAuthorization(**kwargs)

    def _ClusterNetworking(self, args):
        kwargs = {'podAddressCidrBlocks': flags.GetPodAddressCidrBlocks(args), 'serviceAddressCidrBlocks': flags.GetServiceAddressCidrBlocks(args), 'serviceLoadBalancerSubnetId': flags.GetServiceLoadBalancerSubnetId(args), 'virtualNetworkId': flags.GetVnetId(args)}
        if not any(kwargs.values()):
            return None
        return self._messages.GoogleCloudGkemulticloudV1AzureClusterNetworking(**kwargs)

    def _ControlPlane(self, args):
        control_plane_type = self._messages.GoogleCloudGkemulticloudV1AzureControlPlane
        kwargs = {'configEncryption': self._ConfigEncryption(args), 'databaseEncryption': self._DatabaseEncryption(args), 'endpointSubnetId': flags.GetEndpointSubnetId(args), 'mainVolume': self._DiskTemplate(args, 'main'), 'proxyConfig': self._ProxyConfig(args), 'replicaPlacements': flags.GetReplicaPlacements(args), 'rootVolume': self._DiskTemplate(args, 'root'), 'sshConfig': self._SshConfig(args), 'subnetId': flags.GetSubnetID(args), 'tags': self._Tags(args, control_plane_type), 'version': flags.GetClusterVersion(args), 'vmSize': flags.GetVMSize(args)}
        return self._messages.GoogleCloudGkemulticloudV1AzureControlPlane(**kwargs) if any(kwargs.values()) else None

    def _SshConfig(self, args):
        kwargs = {'authorizedKey': flags.GetSSHPublicKey(args)}
        return self._messages.GoogleCloudGkemulticloudV1AzureSshConfig(**kwargs) if any(kwargs.values()) else None

    def _DatabaseEncryption(self, args):
        kwargs = {'keyId': flags.GetDatabaseEncryptionKeyId(args)}
        if not any(kwargs.values()):
            return None
        return self._messages.GoogleCloudGkemulticloudV1AzureDatabaseEncryption(**kwargs)

    def _Autoscaling(self, args):
        kwargs = {'minNodeCount': flags.GetMinNodes(args), 'maxNodeCount': flags.GetMaxNodes(args)}
        if not any(kwargs.values()):
            return None
        return self._messages.GoogleCloudGkemulticloudV1AzureNodePoolAutoscaling(**kwargs)

    def _NodeConfig(self, args):
        node_config_type = self._messages.GoogleCloudGkemulticloudV1AzureNodeConfig
        kwargs = {'configEncryption': self._ConfigEncryption(args), 'imageType': flags.GetImageType(args), 'labels': self._Labels(args, node_config_type), 'proxyConfig': self._ProxyConfig(args), 'rootVolume': self._DiskTemplate(args, 'root'), 'sshConfig': self._SshConfig(args), 'tags': self._Tags(args, node_config_type), 'taints': flags.GetNodeTaints(args), 'vmSize': flags.GetVMSize(args)}
        return self._messages.GoogleCloudGkemulticloudV1AzureNodeConfig(**kwargs) if any(kwargs.values()) else None

    def _NodeManagement(self, args):
        kwargs = {'autoRepair': flags.GetAutoRepair(args)}
        return self._messages.GoogleCloudGkemulticloudV1AzureNodeManagement(**kwargs) if any(kwargs.values()) else None