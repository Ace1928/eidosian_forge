from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six.moves.urllib.parse
def ValidateGpuClass(gpu_class, mode):
    """Validates the gpu_class input.

  Args:
    gpu_class: String indicating the GPU class of the instance. Allowed values
      are l4 and t4.
    mode: String indicating the rendering mode of the instance.

  Returns:
    True if the GPU class and mode combination is supported by ISXR, False
    otherwise.
  """
    gpu_class = gpu_class.lower()
    if gpu_class == 't4':
        return True
    if gpu_class == 'l4':
        if not mode or mode.lower() != '3d':
            raise exceptions.InvalidArgumentException('--gpu-class', 'l4 gpu-class must have --mode=3d')
        return True
    raise exceptions.InvalidArgumentException('--gpu-class', 'gpu-class must be l4 or t4')