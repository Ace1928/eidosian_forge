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
def RetrieveKubernetesRootCert(self):
    """Retrieves the root cert from the cluster."""
    out, err = self._RunKubectl(['get', 'configmap', 'kube-root-ca.crt', '-o', 'jsonpath="{.data.ca\\.crt}"'], None)
    if err:
        if 'NotFound' in err:
            raise ClusterError('Cluster root certificate is not found.')
        raise exceptions.Error('Error retrieving Kubernetes root cert: {}'.format(err))
    return out.strip('"')