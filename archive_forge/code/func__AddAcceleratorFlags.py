from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_util
from googlecloudsdk.command_lib.util.args import labels_util
def _AddAcceleratorFlags(parser):
    """Add arguments for accelerator config."""
    accelerator_config_group = base.ArgumentGroup(help='Accelerator Configuration.')
    accelerator_config_group.AddArgument(base.Argument('--accelerator-count', required=True, default=1, type=arg_parsers.BoundedInt(lower_bound=1), help='The number of accelerators to attach to the machines. Must be >= 1.'))
    accelerator_config_group.AddArgument(jobs_util.AcceleratorFlagMap().choice_arg)
    accelerator_config_group.AddToParser(parser)