from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
@staticmethod
def _GkeNodePoolAutoscalingConfigFromArgPool(dataproc, arg_pool):
    """Creates the GkeNodePoolAutoscalingConfig via the arguments specified in --pools."""
    config = dataproc.messages.GkeNodePoolAutoscalingConfig()
    if 'min' in arg_pool:
        config.minNodeCount = arg_pool['min']
    if 'max' in arg_pool:
        config.maxNodeCount = arg_pool['max']
    if config != dataproc.messages.GkeNodePoolAutoscalingConfig():
        return config
    return None