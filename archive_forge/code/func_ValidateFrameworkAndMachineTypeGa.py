from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.command_lib.ml_engine import models_util
from googlecloudsdk.command_lib.ml_engine import uploads
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def ValidateFrameworkAndMachineTypeGa(framework, machine_type):
    frameworks_enum = versions_api.GetMessagesModule().GoogleCloudMlV1Version.FrameworkValueValuesEnum
    if framework != frameworks_enum.TENSORFLOW and (not machine_type.startswith('ml')):
        raise InvalidArgumentCombinationError('Machine type {0} is currently only supported with tensorflow.'.format(machine_type))