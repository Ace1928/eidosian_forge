from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances import utils as instances_utils
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
def GetProjectToServiceAccountMap(args, instance_refs, client, skip_defaults):
    """Creates a mapping of projects to service accounts."""
    project_to_sa = {}
    for instance_ref in instance_refs:
        if instance_ref.project not in project_to_sa:
            project_to_sa[instance_ref.project] = GetProjectServiceAccount(args=args, project=instance_ref.project, client=client, skip_defaults=skip_defaults, instance_name=instance_ref.Name())
    return project_to_sa