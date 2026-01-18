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
def RetrieveASMVersion(self, revision):
    """Retrieves the version of ASM."""
    image, err = self._RunKubectl(['get', 'deploy', '-l', 'istio.io/rev={},app=istiod'.format(revision), '-n', 'istio-system', '-o', 'jsonpath="{.items[0].spec.template.spec.containers[0].image}"'], None)
    if err:
        if 'NotFound' in err:
            raise ClusterError('Anthos Service Mesh revision {} is not found in the cluster. Please install Anthos Service Mesh and try again.'.format(revision))
        raise exceptions.Error('Error retrieving the version of Anthos Service Mesh: {}'.format(err))
    if not image:
        raise ClusterError('Anthos Service Mesh revision {} does not have an image property. Please re-install Anthos Service Mesh.'.format(revision))
    version_match = re.search(_ASM_VERSION_PATTERN, image)
    if version_match:
        return version_match.group(1)
    raise exceptions.Error('Value image: {} is invalid.'.format(image))