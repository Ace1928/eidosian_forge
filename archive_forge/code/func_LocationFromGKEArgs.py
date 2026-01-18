from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import re
import textwrap
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def LocationFromGKEArgs(args):
    """Returns the location for a membership based on GKE cluster flags.

  For GKE clusters, use cluster location as membership location, unless
  they are registered with kubeconfig in which case they are not
  considered "GKE clusters."

  Args:
    args: The command line args

  Returns:
    a location, e.g. "global" or "us-central1".

  Raises:
    a core.Error, if the location could not be found in the flag
  """
    location = ''
    if args.gke_cluster:
        location_re = re.search('([a-z0-9]+\\-[a-z0-9]+)(\\-[a-z])?/(\\-[a-z])?', args.gke_cluster)
        if location_re:
            location = location_re.group(1)
        else:
            raise exceptions.Error('Unable to parse location from `gke-cluster` parameter. Expecting `$CLUSTER_LOCATION/$CLUSTER_NAME` e.g. `us-central1/my-cluster`')
    elif args.gke_uri:
        location_re = re.search('(regions|locations|zones)/([a-z0-9]+\\-[a-z0-9]+)(\\-[a-z])?/clusters', args.gke_uri)
        if location_re:
            location = location_re.group(2)
        else:
            raise exceptions.Error('Unable to parse location from `gke-uri` parameter. Expecting a string like projects/123/locations/us-central1-a/clusters/my-cluster')
    return location