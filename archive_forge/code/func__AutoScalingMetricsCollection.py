from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.aws import flags as aws_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _AutoScalingMetricsCollection(self, args):
    kwargs = {'granularity': aws_flags.GetAutoscalingMetricsGranularity(args), 'metrics': aws_flags.GetAutoscalingMetrics(args)}
    if not any(kwargs.values()):
        return None
    return self._messages.GoogleCloudGkemulticloudV1AwsAutoscalingGroupMetricsCollection(**kwargs)