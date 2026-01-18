from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateEnvironment(args, messages):
    """Creates the Environment message for the create request.

  Args:
    args: Argparse object from Command.Run
    messages: Module containing messages definition for the specified API.

  Returns:
    Instance of the Environment message.
  """

    def CreateContainerImageFromArgs(args):
        container_image = messages.ContainerImage(repository=args.container_repository, tag=args.container_tag)
        return container_image

    def CreateVmImageFromArgs(args):
        vm_image = messages.VmImage(project=args.vm_image_project)
        if args.IsSpecified('vm_image_family'):
            vm_image.imageFamily = args.vm_image_family
        else:
            vm_image.imageName = args.vm_image_name
        return vm_image
    if args.IsSpecified('vm_image_project'):
        vm_image = CreateVmImageFromArgs(args)
    else:
        container_image = CreateContainerImageFromArgs(args)
    environment = messages.Environment(name=args.environment, description=args.description, displayName=args.display_name, postStartupScript=args.post_startup_script)
    if args.IsSpecified('vm_image_project'):
        environment.vmImage = vm_image
    else:
        environment.containerImage = container_image
    return environment