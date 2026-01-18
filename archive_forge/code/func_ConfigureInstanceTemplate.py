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
def ConfigureInstanceTemplate(args, kube_client, project_id, network_resource, workload_namespace, workload_name, workload_manifest, membership_manifest, asm_revision, mesh_config):
    """Configure the provided instance template args with ASM metadata."""
    is_mcp = _IsMCP(kube_client, asm_revision)
    service_proxy_metadata_args = _RetrieveServiceProxyMetadata(args, is_mcp, kube_client, project_id, network_resource, workload_namespace, workload_name, workload_manifest, membership_manifest, asm_revision, mesh_config)
    _ModifyInstanceTemplate(args, is_mcp, service_proxy_metadata_args)