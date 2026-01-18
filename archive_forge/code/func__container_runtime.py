from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Optional
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _container_runtime(self, container_runtime):
    """Constructs proto message BareMetalWorkloadNodeConfig.ContainerRuntimeValueValuesEnum."""
    if container_runtime is None:
        return None
    container_runtime_enum = messages.BareMetalWorkloadNodeConfig.ContainerRuntimeValueValuesEnum
    container_runtime_mapping = {'ContainerRuntimeUnspecified': container_runtime_enum.CONTAINER_RUNTIME_UNSPECIFIED, 'Conatinerd': container_runtime_enum.CONTAINERD}
    return container_runtime_mapping[container_runtime]