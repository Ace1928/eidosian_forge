from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from typing import Iterator, List
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as fleet_messages
def _DefaultClusterConfig(self, existing_default_cluster_config=None) -> fleet_messages.DefaultClusterConfig:
    ret = self.messages.DefaultClusterConfig()
    ret.securityPostureConfig = self._SecurityPostureConfig()
    if existing_default_cluster_config is not None:
        ret.binaryAuthorizationConfig = self._BinaryAuthorizationConfig(existing_default_cluster_config.binaryAuthorizationConfig)
    else:
        ret.binaryAuthorizationConfig = self._BinaryAuthorizationConfig()
    return self.TrimEmpty(ret)