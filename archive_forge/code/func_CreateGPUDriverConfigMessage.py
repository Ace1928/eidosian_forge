from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateGPUDriverConfigMessage(args, messages):
    if not (args.IsSpecified('custom_gpu_driver_path') or args.IsSpecified('install_gpu_driver')):
        return None
    return messages.GPUDriverConfig(customGpuDriverPath=args.custom_gpu_driver_path, enableGpuDriver=args.install_gpu_driver)