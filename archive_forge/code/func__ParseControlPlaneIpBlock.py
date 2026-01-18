from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ParseControlPlaneIpBlock(value):
    """Parse the given value in IP block format.

  Args:
    value: str, supports format of either IP, or 'IP hostname'.

  Returns:
    tuple: of structure (IP, hostname).

  Raises:
    exceptions.InvalidArgumentException: raise parsing error.
  """
    parsing_error = "Malformed IP block [{}].\nExpect an individual IP address, or an individual IP address with an optional hostname.\nExamples: --control-plane-ip-block 'netmask=255.255.255.0,gateway=10.0.0.0,ips=192.168.1.1;0.0.0.0 localhost'\n".format(value)
    if ' ' not in value:
        return (value, None)
    else:
        ip_block = value.split(' ')
        if len(ip_block) != 2:
            raise exceptions.InvalidArgumentException('--control-plane-ip-block', message=parsing_error)
        else:
            return (ip_block[0], ip_block[1])