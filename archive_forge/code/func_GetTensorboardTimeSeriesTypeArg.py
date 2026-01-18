from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.tensorboard_time_series import client
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import resources
def GetTensorboardTimeSeriesTypeArg(noun):
    return arg_utils.ChoiceEnumMapper('--type', client.GetMessagesModule().GoogleCloudAiplatformV1beta1TensorboardTimeSeries.ValueTypeValueValuesEnum, required=True, custom_mappings=_TYPE_CHOICES, help_str='Value type of the {noun}.'.format(noun=noun), default=None)