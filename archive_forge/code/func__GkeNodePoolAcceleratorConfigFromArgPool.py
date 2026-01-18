from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
@staticmethod
def _GkeNodePoolAcceleratorConfigFromArgPool(dataproc, arg_accelerators):
    """Creates the GkeNodePoolAcceleratorConfig via the arguments specified in --pools."""
    accelerators = []
    for arg_accelerator in arg_accelerators.split(';'):
        if '=' not in arg_accelerator:
            raise exceptions.InvalidArgumentException('--pools', 'accelerators value "%s" does not match the expected "ACCELERATOR_TYPE=ACCELERATOR_VALUE" pattern.' % arg_accelerator)
        accelerator_type, count_string = arg_accelerator.split('=', 1)
        try:
            count = int(count_string)
            accelerators.append(dataproc.messages.GkeNodePoolAcceleratorConfig(acceleratorCount=count, acceleratorType=accelerator_type))
        except ValueError:
            raise exceptions.InvalidArgumentException('--pools', 'Unable to parse accelerators count "%s" as an integer.' % count_string)
    return accelerators