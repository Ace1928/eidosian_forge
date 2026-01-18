from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def ParseReplaceAccessLevelsResponse(api_version):
    """Wrapper around ParseReplaceAccessLevelsResponse to accept api version."""

    def VersionedParseReplaceAccessLevelsResponse(lro, unused_args):
        """Parse the Long Running Operation response of the ReplaceAccessLevels call.

    Args:
      lro: Long Running Operation response of ReplaceAccessLevels.
      unused_args: not used.

    Returns:
      The replacement Access Levels created by the ReplaceAccessLevels call.

    Raises:
      ParseResponseError: if the response could not be parsed into the proper
      object.
    """
        client = util.GetClient(version=api_version)
        operation_ref = resources.REGISTRY.Parse(lro.name, collection='accesscontextmanager.operations')
        poller = common.BulkAPIOperationPoller(client.accessPolicies_accessLevels, client.operations, operation_ref)
        return waiter.WaitFor(poller, operation_ref, 'Waiting for Replace Access Levels operation [{}]'.format(operation_ref.Name()))
    return VersionedParseReplaceAccessLevelsResponse