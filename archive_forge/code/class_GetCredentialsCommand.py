from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import List
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.container import util as container_util
from googlecloudsdk.api_lib.container.fleet.connectgateway import client as gateway_client
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util as hubapi_util
from googlecloudsdk.command_lib.container.fleet import base as hub_base
from googlecloudsdk.command_lib.container.fleet import gwkubeconfig_util as kconfig
from googlecloudsdk.command_lib.container.fleet import overrides
from googlecloudsdk.command_lib.container.fleet.memberships import errors as memberships_errors
from googlecloudsdk.command_lib.container.fleet.memberships import util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class GetCredentialsCommand(hub_base.HubCommand, base.Command):
    """GetCredentialsCommand is a base class with util functions for Gateway credential generating commands."""

    def RunGetCredentials(self, membership_id, arg_location, arg_namespace=None):
        container_util.CheckKubectlInstalled()
        project_id = hub_base.HubCommand.Project()
        log.status.Print('Starting to build Gateway kubeconfig...')
        log.status.Print('Current project_id: ' + project_id)
        self.RunIamCheck(project_id, REQUIRED_CLIENT_PERMISSIONS)
        try:
            hub_endpoint_override = properties.VALUES.api_endpoint_overrides.Property('gkehub').Get()
        except properties.NoSuchPropertyError:
            hub_endpoint_override = None
        CheckGatewayApiEnablement(project_id, util.GetConnectGatewayServiceName(hub_endpoint_override, None))
        membership = self.ReadClusterMembership(project_id, arg_location, membership_id)
        collection = 'memberships'
        if project_id == 'gkeconnect-prober':
            pass
        elif hasattr(membership, 'endpoint') and hasattr(membership.endpoint, 'gkeCluster') and membership.endpoint.gkeCluster:
            collection = 'gkeMemberships'
        self.GenerateKubeconfig(util.GetConnectGatewayServiceName(hub_endpoint_override, arg_location), project_id, arg_location, collection, membership_id, arg_namespace)
        msg = 'A new kubeconfig entry "' + self.KubeContext(project_id, arg_location, membership_id, arg_namespace) + '" has been generated and set as the current context.'
        log.status.Print(msg)

    def RunServerSide(self, membership_id: str, arg_location: str, force_use_agent: bool=False):
        """RunServerSide generates credentials using server-side kubeconfig generation.

    Args:
      membership_id: The short name of the membership to generate credentials
        for.
      arg_location: The location of the membership to generate credentials for.
      force_use_agent: Whether to force the use of Connect Agent in generated
        credentials.
    """
        log.status.Print('Fetching Gateway kubeconfig...')
        container_util.CheckKubectlInstalled()
        project_id = hub_base.HubCommand.Project()
        project_number = hub_base.HubCommand.Project(number=True)
        self.RunIamCheck(project_id, REQUIRED_SERVER_PERMISSIONS)
        with overrides.RegionalGatewayEndpoint(arg_location):
            client = gateway_client.GatewayClient(self.ReleaseTrack())
            resp = client.GenerateCredentials(name=f'projects/{project_number}/locations/{arg_location}/memberships/{membership_id}', force_use_agent=force_use_agent)
        new = kconfig.Kubeconfig.LoadFromBytes(resp.kubeconfig)
        kubeconfig = kconfig.Kubeconfig.Default()
        kubeconfig.Merge(new, overwrite=True)
        kubeconfig.SetCurrentContext(list(new.contexts.keys())[0])
        kubeconfig.SaveToFile()
        msg = f'A new kubeconfig entry "{kubeconfig.current_context}" has been generated and set as the current context.'
        log.status.Print(msg)

    def KubeContext(self, project_id, location, membership, namespace=None):
        kc = KUBECONTEXT_FORMAT.format(project=project_id, location=location, membership=membership)
        if namespace:
            kc += '_ns-' + namespace
        return kc

    def RunIamCheck(self, project_id: str, permissions: List[str]):
        """Run an IAM check, making sure the caller has the necessary permissions to use the Gateway API."""
        project_ref = project_util.ParseProject(project_id)
        result = projects_api.TestIamPermissions(project_ref, permissions)
        granted_permissions = result.permissions
        if not set(permissions).issubset(set(granted_permissions)):
            raise memberships_errors.InsufficientPermissionsError()

    def ReadClusterMembership(self, project_id, location, membership):
        resource_name = hubapi_util.MembershipRef(project_id, location, membership)
        return hubapi_util.GetMembership(resource_name)

    def GenerateKubeconfig(self, service_name, project_id, location, collection, membership, namespace=None):
        project_number = project_util.GetProjectNumber(project_id)
        kwargs = {'membership': membership, 'location': location, 'project_id': project_id, 'server': SERVER_FORMAT.format(service_name=service_name, version=self.GetVersion(), project_number=project_number, location=location, collection=collection, membership=membership), 'auth_provider': 'gcp'}
        user_kwargs = {'auth_provider': 'gcp'}
        cluster_kwargs = {}
        context = self.KubeContext(project_id, location, membership, namespace)
        cluster = self.KubeContext(project_id, location, membership)
        kubeconfig = kconfig.Kubeconfig.Default()
        kubeconfig.contexts[context] = kconfig.Context(context, cluster, context, namespace)
        kubeconfig.users[context] = kconfig.User(context, **user_kwargs)
        kubeconfig.clusters[cluster] = kconfig.Cluster(cluster, kwargs['server'], **cluster_kwargs)
        kubeconfig.SetCurrentContext(context)
        kubeconfig.SaveToFile()
        return kubeconfig

    @classmethod
    def GetVersion(cls):
        if cls.ReleaseTrack() is base.ReleaseTrack.ALPHA:
            return 'v1alpha1'
        elif cls.ReleaseTrack() is base.ReleaseTrack.BETA:
            return 'v1beta1'
        elif cls.ReleaseTrack() is base.ReleaseTrack.GA:
            return 'v1'
        else:
            return ''