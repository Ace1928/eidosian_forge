from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.os_config import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
def SetParentOnListRequestHook(unused_ref, args, request):
    """Add parent field to List request.

  Args:
    unused_ref: A parsed resource reference; unused.
    args: The parsed args namespace from CLI
    request: List request for the API call

  Returns:
    Modified request that includes the parent field.
  """
    project = args.project or properties.VALUES.core.project.GetOrFail()
    location = args.location or properties.VALUES.compute.zone.Get()
    if not location:
        raise exceptions.Error('Location value is required either from `--location` or default zone, see {url}. '.format(url='https://cloud.google.com/compute/docs/gcloud-compute#default-region-zone'))
    instance = args.instance or '-'
    os_policy_assignment = args.assignment_id or '-'
    flags.ValidateInstance(instance, '--instance')
    flags.ValidateZone(location, '--location')
    flags.ValidateInstanceOsPolicyAssignment(os_policy_assignment, '--assignment-id')
    request.parent = _LIST_URI.format(project=project, location=location, instance=instance, os_policy_assignment=os_policy_assignment)
    return request