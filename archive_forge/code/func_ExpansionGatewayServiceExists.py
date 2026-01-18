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
def ExpansionGatewayServiceExists(self):
    """Verifies if the ASM Expansion Gateway service exists."""
    _, err = self._RunKubectl(['get', 'service', _EXPANSION_GATEWAY_NAME, '-n', 'istio-system'], None)
    if err:
        if 'NotFound' in err:
            return False
        raise exceptions.Error('Error retrieving the expansion gateway service: {}'.format(err))
    return True