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
def AddDeviceRegistrySettingsFlagsToParser(parser, defaults=True):
    """Get flags for device registry commands.

  Args:
    parser: argparse parser to which to add these flags.
    defaults: bool, whether to populate default values (for instance, should be
        false for Patch commands).

  Returns:
    list of base.Argument, the flags common to and specific to device commands.
  """
    base.Argument('--enable-mqtt-config', help='Whether to allow MQTT connections to this device registry.', default=True if defaults else None, action='store_true').AddToParser(parser)
    base.Argument('--enable-http-config', help='Whether to allow device connections to the HTTP bridge.', default=True if defaults else None, action='store_true').AddToParser(parser)
    base.Argument('--state-pubsub-topic', required=False, help='A Google Cloud Pub/Sub topic name for state notifications.').AddToParser(parser)
    for f in _GetEventNotificationConfigFlags():
        f.AddToParser(parser)