from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from six.moves import map  # pylint: disable=redefined-builtin
from the device. This flag can be specified multiple times to add multiple
def _GetCreateFlagsForGateways():
    """Generates all the flags for the create command."""
    return _GetDeviceFlags() + _GetDeviceCredentialFlags() + [CREATE_GATEWAY_ENUM_MAPPER.choice_arg, GATEWAY_AUTH_METHOD_ENUM_MAPPER.choice_arg]