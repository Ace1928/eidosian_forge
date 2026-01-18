from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.core import yaml
def GetAlgorithmEnum(version=constants.BETA_VERSION):
    messages = apis.GetMessagesModule(constants.AI_PLATFORM_API_NAME, constants.AI_PLATFORM_API_VERSION[version])
    if version == constants.GA_VERSION:
        return messages.GoogleCloudAiplatformV1StudySpec.AlgorithmValueValuesEnum
    else:
        return messages.GoogleCloudAiplatformV1beta1StudySpec.AlgorithmValueValuesEnum