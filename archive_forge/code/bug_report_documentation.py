from googlecloudsdk.api_lib.container.fleet import debug_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.mesh import istioctl_backend
from googlecloudsdk.core import properties
Capture cluster information and logs into archive to help diagnose problems.

  Example: ${command} --project projectId
                      --membership membershipId
                      --location location
  