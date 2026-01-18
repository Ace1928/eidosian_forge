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
def _ParseMembershipName(owner_id):
    """Get membership name from an owner id value.

  Args:
    owner_id: The owner ID value of a membership. e.g.,
    //gkehub.googleapis.com/projects/123/locations/global/memberships/test.

  Returns:
    The full resource name of the membership, e.g.,
      projects/foo/locations/global/memberships/name.

  Raises:
    Error: if the membership name cannot be parsed.
  """
    membership_match = re.search(_GKEHUB_PATTERN, owner_id)
    if membership_match:
        return membership_match.group(2)
    raise exceptions.Error('value owner_id: {} is invalid.'.format(owner_id))