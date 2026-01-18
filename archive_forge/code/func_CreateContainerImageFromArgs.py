from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateContainerImageFromArgs(args, messages):
    if args.IsSpecified('container_repository'):
        container_image = messages.ContainerImage(repository=args.container_repository, tag=args.container_tag)
        return container_image
    return None