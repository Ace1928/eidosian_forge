from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.tensorboards import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.core import resources
def _GetUriBeta(tensorboard):
    ref = resources.REGISTRY.ParseRelativeName(tensorboard.name, constants.TENSORBOARDS_COLLECTION, api_version=constants.AI_PLATFORM_API_VERSION[constants.BETA_VERSION])
    return ref.SelfLink()