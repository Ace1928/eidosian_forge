from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import re
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util as hub_kube_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _RetrieveServiceProxyMetadata(args, is_mcp, kube_client, project_id, network_resource, workload_namespace, workload_name, workload_manifest, membership_manifest, asm_revision, mesh_config):
    """Retrieve the necessary metadata to configure the service proxy."""
    if is_mcp:
        asm_version = None
        expansionagateway_ip = None
        root_cert = None
        service_proxy_api_server = _RetrieveDiscoveryAddress(mesh_config)
        env_config = kube_client.RetrieveEnvConfig(asm_revision)
    else:
        if _GCE_SERVICE_PROXY_ASM_VERSION_METADATA in args.metadata:
            asm_version = args.metadata[_GCE_SERVICE_PROXY_ASM_VERSION_METADATA]
        else:
            asm_version = kube_client.RetrieveASMVersion(asm_revision)
        expansionagateway_ip = kube_client.RetrieveExpansionGatewayIP()
        root_cert = kube_client.RetrieveKubernetesRootCert()
        service_proxy_api_server = '{}:{}'.format(expansionagateway_ip, _ISTIO_DISCOVERY_PORT)
        env_config = None
    identity_provider = _GetVMIdentityProvider(membership_manifest, workload_namespace)
    service_account = _RetrieveWorkloadServiceAccount(workload_manifest)
    asm_proxy_config = _RetrieveProxyConfig(is_mcp, mesh_config)
    trust_domain = _RetrieveTrustDomain(is_mcp, mesh_config)
    mesh_id = _RetrieveMeshId(is_mcp, mesh_config)
    network = network_resource.split('/')[-1]
    asm_labels = _GetWorkloadLabels(workload_manifest)
    canonical_service = _GetCanonicalServiceName(workload_name, workload_manifest)
    canonical_revision = _GetCanonicalServiceRevision(workload_manifest)
    return ServiceProxyMetadataArgs(asm_version, project_id, expansionagateway_ip, service_proxy_api_server, identity_provider, service_account, asm_proxy_config, env_config, trust_domain, mesh_id, network, asm_labels, workload_name, workload_namespace, canonical_service, canonical_revision, asm_revision, root_cert)