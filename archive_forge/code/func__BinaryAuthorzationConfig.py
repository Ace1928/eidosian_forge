from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from typing import Iterator
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources as fleet_resources
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as fleet_messages
def _BinaryAuthorzationConfig(self) -> fleet_messages.BinaryAuthorizationConfig:
    binary_authorization_config = fleet_messages.BinaryAuthorizationConfig()
    binary_authorization_config.evaluationMode = self._EvaluationMode()
    binary_authorization_config.policyBindings = list(self._PolicyBindings())
    return self.TrimEmpty(binary_authorization_config)