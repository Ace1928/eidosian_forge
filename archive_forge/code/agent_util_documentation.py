from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import gkehub_api_adapter
from googlecloudsdk.api_lib.container.fleet import gkehub_api_util
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.projects import util as p_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
Returns the namespaces into which to install or update the connect agent.

  Connect namespaces are identified by the presence of the hub.gke.io/project
  label. If there are existing namespaces with this label in the cluster,
  then a list of all those namespaces is returned; otherwise, a list with the
  default connect namespace is returned.

  Args:
    kube_client: a KubernetesClient.
    project_id: A GCP project identifier.

  Returns:
    List of namespaces with hub.gke.io/project label.
  