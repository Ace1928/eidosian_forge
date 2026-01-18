from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def _VolumeTypeEnumMapper(prefix):
    return arg_utils.ChoiceEnumMapper('--{}-volume-type'.format(prefix), api_util.GetMessagesModule().GoogleCloudGkemulticloudV1AwsVolumeTemplate.VolumeTypeValueValuesEnum, include_filter=lambda volume_type: 'UNSPECIFIED' not in volume_type, help_str='Type of the {} volume.'.format(prefix))