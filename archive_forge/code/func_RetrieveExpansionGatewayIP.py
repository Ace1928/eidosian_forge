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
def RetrieveExpansionGatewayIP(self):
    """Retrieves the expansion gateway IP from the cluster."""
    if not self.ExpansionGatewayDeploymentExists():
        raise ClusterError('The gateway {} deployment is not found in the cluster. Please install Anthos Service Mesh with VM support and retry.'.format(_EXPANSION_GATEWAY_NAME))
    if not self.ExpansionGatewayServiceExists():
        raise ClusterError('The gateway {} service is not found in the cluster. Please install Anthos Service Mesh with VM support and retry.'.format(_EXPANSION_GATEWAY_NAME))
    out, err = self._RunKubectl(['get', 'svc', _EXPANSION_GATEWAY_NAME, '-n', 'istio-system', '-o', 'jsonpath={.status.loadBalancer.ingress[0].ip}'], None)
    if err:
        raise exceptions.Error('Error retrieving expansion gateway IP: {}'.format(err))
    return out