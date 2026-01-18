from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
import json
import os
import re
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import format_util
from googlecloudsdk.command_lib.container.fleet.memberships import gke_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from kubernetes import client as kube_client_lib
from kubernetes import config as kube_client_config
from six.moves.urllib.parse import urljoin
class KubernetesClient(object):
    """A client for accessing a subset of the Kubernetes API."""

    def __init__(self, api_adapter=None, gke_uri=None, gke_cluster=None, kubeconfig=None, internal_ip=False, cross_connect_subnetwork=None, private_endpoint_fqdn=None, context=None, public_issuer_url=None, enable_workload_identity=False):
        """Constructor for KubernetesClient.

    Args:
      api_adapter: the GKE api adapter used for running kubernetes commands
      gke_uri: the URI of the GKE cluster; for example,
        'https://container.googleapis.com/v1/projects/my-project/locations/us-central1-a/clusters/my-cluster'
      gke_cluster: the "location/name" of the GKE cluster. The location can be a
        zone or a region for e.g `us-central1-a/my-cluster`
      kubeconfig: the kubeconfig path
      internal_ip: whether to persist the internal IP of the endpoint.
      cross_connect_subnetwork: full path of the cross connect subnet whose
        endpoint to persist (optional)
      private_endpoint_fqdn: whether to persist the private fqdn.
      context: the context to use
      public_issuer_url: the public issuer url
      enable_workload_identity: whether to enable workload identity

    Raises:
      exceptions.Error: if the client cannot be configured
      calliope_exceptions.MinimumArgumentException: if a kubeconfig file
        cannot be deduced from the command line flags or environment
    """
        self.kubectl_timeout = '20s'
        self.temp_kubeconfig_dir = None
        if gke_uri or gke_cluster:
            self.temp_kubeconfig_dir = files.TemporaryDirectory()
        self.processor = KubeconfigProcessor(api_adapter=api_adapter, gke_uri=gke_uri, gke_cluster=gke_cluster, kubeconfig=kubeconfig, internal_ip=internal_ip, cross_connect_subnetwork=cross_connect_subnetwork, private_endpoint_fqdn=private_endpoint_fqdn, context=context)
        self.kubeconfig, self.context = self.processor.GetKubeconfigAndContext(self.temp_kubeconfig_dir)
        if public_issuer_url or (enable_workload_identity and self.processor.gke_cluster_uri):
            return
        if enable_workload_identity:
            self.kube_client = self.processor.GetKubeClient(self.kubeconfig, self.context)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.temp_kubeconfig_dir is not None:
            self.temp_kubeconfig_dir.Close()

    def CheckClusterAdminPermissions(self):
        """Check to see if the user can perform all the actions in any namespace.

    Raises:
      KubectlError: if failing to get check for cluster-admin permissions.
      RBACError: if cluster-admin permissions are not found.
    """
        out, err = self._RunKubectl(['auth', 'can-i', '*', '*', '--all-namespaces'], None)
        if err:
            raise KubectlError('Failed to check if the user is a cluster-admin: {}'.format(err))
        if 'yes' not in out:
            raise RBACError('Missing cluster-admin RBAC role: The cluster-admin role-based accesscontrol (RBAC) ClusterRole grants you the cluster permissions necessary to connect your clusters back to Google. \nTo create a ClusterRoleBinding resource in the cluster, run the following command:\n\nkubectl create clusterrolebinding [BINDING_NAME]  --clusterrole cluster-admin --user [USER]')

    def GetNamespaceUID(self, namespace):
        out, err = self._RunKubectl(['get', 'namespace', namespace, '-o', "jsonpath='{.metadata.uid}'"], None)
        if err:
            raise exceptions.Error('Failed to get the UID of the cluster: {}'.format(err))
        return out.replace("'", '')

    def GetServerVersion(self):
        """Get server version of the cluster.

    Raises:
      exceptions.Error: if failing to get namespaces.

    Returns:
      Server version of the cluster in major.minor format (e.g. 1.21)
    """
        out, err = self._RunKubectl(['version', '-o', 'json'], None)
        if err:
            raise exceptions.Error('Failed to get the server version of the cluster: {}'.format(err))
        out = json.loads(out)
        version_str = out['serverVersion']['major'] + '.' + out['serverVersion']['minor']
        return version_str

    def GetEvents(self, namespace):
        """Get k8s events for the namespace."""
        out, err = self._RunKubectl(['get', 'events', '--namespace=' + namespace, "--sort-by='{.lastTimestamp}'"], None)
        if err:
            raise exceptions.Error()
        return out

    def NamespacesWithLabelSelector(self, label):
        """Get the Connect Agent namespace by label.

    Args:
      label: the label used for namespace selection

    Raises:
      exceptions.Error: if failing to get namespaces.

    Returns:
      The first namespace with the label selector.
    """
        out, err = self._RunKubectl(['get', 'namespaces', '--selector', label, '-o', 'jsonpath={.items}'], None)
        if err:
            raise exceptions.Error('Failed to list namespaces in the cluster: {}'.format(err))
        if out == '[]':
            return []
        out, err = self._RunKubectl(['get', 'namespaces', '--selector', label, '-o', 'jsonpath={.items[0].metadata.name}'], None)
        if err:
            raise exceptions.Error('Failed to list namespaces in the cluster: {}'.format(err))
        return out.strip().split(' ') if out else []

    def DeleteMembership(self):
        _, err = self._RunKubectl(['delete', 'membership', 'membership'])
        return err

    def GetRbacPermissionPolicy(self, rbac_policy_name, role):
        """Get the RBAC cluster role binding policy."""
        cluster_pattern = re.compile('^clusterrole/')
        namespace_pattern = re.compile('^role/')
        if cluster_pattern.match(role.lower()):
            out, error = self._RunKubectl(['get', 'clusterrolebinding', rbac_policy_name, '-o', 'yaml'])
            if error:
                raise exceptions.Error('Error retrieving RBAC policy: {}'.format(rbac_policy_name))
            return out
        if namespace_pattern.match(role.lower()):
            out, error = self._RunKubectl(['get', 'rolebinding', rbac_policy_name, '-o', 'yaml'])
            if error:
                raise exceptions.Error('Error retrieving RBAC policy: {}'.format(rbac_policy_name))
            return out

    def CleanUpRbacPolicy(self, rbac_to_check):
        """Clean up the RBAC cluster role binding policy."""
        for rbac_policy_pair in rbac_to_check:
            rbac_type = rbac_policy_pair[0]
            rbac_name = rbac_policy_pair[1]
            out, err = self._RunKubectl(['delete', rbac_type, rbac_name], None)
            if err:
                if 'NotFound' in err:
                    log.status.Print('{} for RBAC policy: {} not exist.'.format(rbac_type, rbac_name))
                    continue
                else:
                    raise exceptions.Error('Error deleting RBAC policy: {}'.format(err))
            else:
                log.status.Print('{}'.format(out))
        return True

    def GetRbacPolicyDiff(self, rbac_policy_file):
        out, err = self._RunKubectlDiff(['diff', '-f', rbac_policy_file], None)
        return (out, err)

    def GetRbacPolicy(self, rbac_to_check):
        """Get the RBAC cluster role binding policy."""
        not_found = False
        for rbac_policy_pair in rbac_to_check:
            rbac_type = rbac_policy_pair[0]
            rbac_name = rbac_policy_pair[1]
            _, err = self._RunKubectl(['get', rbac_type, rbac_name])
            if err:
                if 'NotFound' in err:
                    not_found = True
                else:
                    raise exceptions.Error('Error retrieving RBAC policy: {}'.format(err))
            else:
                return False
        if not_found:
            return True

    def GetRBACForOperations(self, membership, role, project_id, identity, is_user, anthos_support):
        """Get the formatted RBAC policy names."""
        cluster_pattern = re.compile('^clusterrole/')
        namespace_pattern = re.compile('^role/')
        rbac_to_check = []
        if is_user:
            rbac_to_check.extend([('clusterrole', format_util.RbacPolicyName('impersonate', project_id, membership, identity, is_user)), ('clusterrolebinding', format_util.RbacPolicyName('impersonate', project_id, membership, identity, is_user))])
        if anthos_support:
            rbac_to_check.append(('clusterrolebinding', format_util.RbacPolicyName('anthos', project_id, membership, identity, is_user)))
        elif cluster_pattern.match(role.lower()):
            rbac_to_check.append(('clusterrolebinding', format_util.RbacPolicyName('permission', project_id, membership, identity, is_user)))
        elif namespace_pattern.match(role.lower()):
            rbac_to_check.append(('rolebinding', format_util.RbacPolicyName('permission', project_id, membership, identity, is_user)))
        return rbac_to_check

    def MembershipCRDExists(self):
        """Returns a boolean indicating if the Membership CRD exists."""
        _, err = self._RunKubectl(['get', 'customresourcedefinitions.apiextensions.k8s.io', 'memberships.hub.gke.io'], None)
        if err:
            if 'NotFound' in err:
                return False
            raise exceptions.Error('Error retrieving Membership CRD: {}'.format(err))
        return True

    def GetMembershipCR(self):
        """Get the YAML representation of the Membership CR."""
        out, err = self._RunKubectl(['get', 'membership', 'membership', '-o', 'yaml'], None)
        if err:
            if 'NotFound' in err:
                return ''
            raise exceptions.Error('Error retrieving membership CR: {}'.format(err))
        return out

    def GetMembershipCRD(self):
        """Get the YAML representation of the Membership CRD."""
        out, err = self._RunKubectl(['get', 'customresourcedefinitions.apiextensions.k8s.io', 'memberships.hub.gke.io', '-o', 'yaml'], None)
        if err:
            if 'NotFound' in err:
                return ''
            raise exceptions.Error('Error retrieving membership CRD: {}'.format(err))
        return out

    def GetMembershipOwnerID(self):
        """Looks up the owner id field in the Membership resource."""
        if not self.MembershipCRDExists():
            return None
        out, err = self._RunKubectl(['get', 'membership', 'membership', '-o', 'jsonpath={.spec.owner.id}'], None)
        if err:
            if 'NotFound' in err:
                return None
            raise exceptions.Error('Error retrieving membership id: {}'.format(err))
        return out

    def CreateMembershipCRD(self, membership_crd_manifest):
        return self.Apply(membership_crd_manifest)

    def ApplyMembership(self, membership_crd_manifest, membership_cr_manifest):
        """Apply membership resources."""
        if membership_crd_manifest:
            _, error = waiter.WaitFor(KubernetesPoller(), MembershipCRDCreationOperation(self, membership_crd_manifest), pre_start_sleep_ms=NAMESPACE_DELETION_INITIAL_WAIT_MS, max_wait_ms=NAMESPACE_DELETION_TIMEOUT_MS, wait_ceiling_ms=NAMESPACE_DELETION_MAX_POLL_INTERVAL_MS, sleep_ms=NAMESPACE_DELETION_INITIAL_POLL_INTERVAL_MS)
            if error:
                raise exceptions.Error('Membership CRD creation failed to complete: {}'.format(error))
        if membership_cr_manifest:
            _, err = self.Apply(membership_cr_manifest)
            if err:
                raise exceptions.Error('Failed to apply Membership CR to cluster: {}'.format(err))

    def ApplyRbacPolicy(self, rbac_policy_file):
        """Applying RBAC policy to Cluster."""
        _, err = self.ApplyRbac(rbac_policy_file)
        if err:
            raise exceptions.Error('Failed to apply rbac policy file to cluster: {}'.format(err))

    def NamespaceExists(self, namespace):
        _, err = self._RunKubectl(['get', 'namespace', namespace])
        return err is None

    def DeleteNamespace(self, namespace):
        _, err = self._RunKubectl(['delete', 'namespace', namespace], timeout_flag='--timeout')
        return err

    def GetResourceField(self, namespace, resource, json_path):
        """Returns the value of a field on a Kubernetes resource.

    Args:
      namespace: the namespace of the resource, or None if this resource is
        cluster-scoped
      resource: the resource, in the format <resourceType>/<name>; e.g.,
        'configmap/foo', or <resourceType> for a list of resources
      json_path: the JSONPath expression to filter with

    Returns:
      The field value (which could be empty if there is no such field), or
      the error printed by the command if there is an error.
    """
        cmd = ['-n', namespace] if namespace else []
        cmd.extend(['get', resource, '-o', 'jsonpath={{{}}}'.format(json_path)])
        return self._RunKubectl(cmd)

    def ApplyRbac(self, rbac_policy):
        out, err = self._RunKubectl(['apply', '-f', rbac_policy], None)
        return (out, err)

    def Apply(self, manifest):
        out, err = self._RunKubectl(['apply', '-f', '-'], stdin=manifest)
        return (out, err)

    def Delete(self, manifest):
        _, err = self._RunKubectl(['delete', '-f', '-'], stdin=manifest)
        return err

    def Logs(self, namespace, log_target):
        """Gets logs from a workload in the cluster.

    Args:
      namespace: the namespace from which to collect logs.
      log_target: the target for the logs command. Any target supported by
        'kubectl logs' is supported here.

    Returns:
      The logs, or an error if there was an error gathering these logs.
    """
        return self._RunKubectl(['logs', '-n', namespace, log_target])

    def _WebRequest(self, method, url, headers=None):
        """Internal method to make requests against web URLs.

    Args:
      method: request method, e.g. GET
      url: request URL
      headers: dictionary of request headers

    Returns:
      Response body as a string

    Raises:
      Error: If the response has a status code >= 400.
    """
        r = requests.GetSession().request(method, url, headers=headers)
        status = r.status_code
        if status >= 400:
            raise exceptions.Error('status: {}, reason: {}'.format(status, r.reason))
        return r.content

    def _ClusterRequest(self, method, api_path, headers=None):
        """Internal method to make requests against the target cluster.

    Args:
      method: request method, e.g. GET
      api_path: path to request against the API server
      headers: dictionary of request headers

    Returns:
      Response body as a string.

    Raises:
      Error: If the response has a status code >= 400.
    """
        if headers is None:
            headers = {}
        self.kube_client.update_params_for_auth(headers=headers, querys=None, auth_settings=['BearerToken'])
        url = urljoin(self.kube_client.configuration.host, api_path)
        r = self.kube_client.rest_client.request(method, url, headers=headers)
        return r.data

    def GetOpenIDConfiguration(self, issuer_url=None):
        """Get the OpenID Provider Configuration for the K8s API server.

    Args:
      issuer_url: string, the issuer URL to query for the OpenID Provider
        Configuration. If None, queries the custer's built-in endpoint.

    Returns:
      The JSON response as a string.

    Raises:
      Error: If the query failed.
    """
        headers = {'Content-Type': 'application/json'}
        url = None
        try:
            if issuer_url is not None:
                url = issuer_url.rstrip('/') + '/.well-known/openid-configuration'
                return self._WebRequest('GET', url, headers=headers)
            else:
                url = '/.well-known/openid-configuration'
                return self._ClusterRequest('GET', url, headers=headers)
        except Exception as e:
            raise exceptions.Error('Failed to get OpenID Provider Configuration from {}: {}'.format(url, e))

    def GetOpenIDKeyset(self, jwks_uri=None):
        """Get the JSON Web Key Set for the K8s API server.

    Args:
      jwks_uri: string, the JWKS URI to query for the JSON Web Key Set. If None,
        queries the cluster's built-in endpoint.

    Returns:
      The JSON response as a string.

    Raises:
      Error: If the query failed.
    """
        headers = {'Content-Type': 'application/jwk-set+json'}
        url = None
        try:
            if jwks_uri is not None:
                url = jwks_uri
                return self._WebRequest('GET', url, headers=headers)
            else:
                url = '/openid/v1/jwks'
                return self._ClusterRequest('GET', url, headers=headers)
        except Exception as e:
            raise exceptions.Error('Failed to get JSON Web Key Set from {}: {}'.format(url, e))

    def _RunKubectl(self, args, stdin=None, timeout_flag='--request-timeout'):
        """Runs a kubectl command with the cluster referenced by this client.

    Args:
      args: command line arguments to pass to kubectl
      stdin: text to be passed to kubectl via stdin
      timeout_flag: kubectl command flag used to set timeout

    Returns:
      The contents of stdout if the return code is 0, stderr (or a fabricated
      error if stderr is empty) otherwise
    """
        cmd = [c_util.CheckKubectlInstalled()]
        if self.context:
            cmd.extend(['--context', self.context])
        if self.kubeconfig:
            cmd.extend(['--kubeconfig', self.kubeconfig])
        cmd.extend([timeout_flag, self.kubectl_timeout])
        cmd.extend(args)
        out = io.StringIO()
        err = io.StringIO()
        returncode = execution_utils.Exec(cmd, no_exit=True, out_func=out.write, err_func=err.write, in_str=stdin)
        if returncode != 0 and (not err.getvalue()):
            err.write('kubectl exited with return code {}'.format(returncode))
        return (out.getvalue() if returncode == 0 else None, err.getvalue() if returncode != 0 else None)

    def _RunKubectlDiff(self, args, stdin=None):
        """Runs a kubectl diff command with the specified args.

    Args:
      args: command line arguments to pass to kubectl
      stdin: text to be passed to kubectl via stdin

    Returns:
      The contents of stdout if the return code is 1, stderr (or a fabricated
      error if stderr is empty) otherwise
    """
        cmd = [c_util.CheckKubectlInstalled()]
        if self.context:
            cmd.extend(['--context', self.context])
        if self.kubeconfig:
            cmd.extend(['--kubeconfig', self.kubeconfig])
        cmd.extend(['--request-timeout', self.kubectl_timeout])
        cmd.extend(args)
        out = io.StringIO()
        err = io.StringIO()
        returncode = execution_utils.Exec(cmd, no_exit=True, out_func=out.write, err_func=err.write, in_str=stdin)
        return (out.getvalue() if returncode == 1 else None, err.getvalue() if returncode > 1 else None)