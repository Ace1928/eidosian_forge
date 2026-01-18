from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.tensorboard_time_series import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ReadBeta(base.Command):
    """Read the Tensorboard time series data from the given Tensorboard time series id."""
    detailed_help = {'EXAMPLES': '          To read Tensorboard Time Series `123` data in Tensorboard `12345`, Tensorboard Experiment `my-tensorboard-experiment, Tensorboard Run `my-tensorboard-run`, region `us-central1`, and project `my-project`:\n\n              $ {command} projects/my-project/locations/us-central1/tensorboards/12345/experiments/my-tensorboard-experiment/runs/my-tensorboard-run/timeSeries/123\n\n          Or with flags:\n\n              $ {command} 123 --tensorboard-id=12345 --tensorboard-experiment-id=my-tensorboard-experiment --tensorboard-run-id=my-tensorboard-run\n          '}

    @staticmethod
    def Args(parser):
        _AddArgs(parser)

    def Run(self, args):
        return _Run(args, constants.BETA_VERSION)