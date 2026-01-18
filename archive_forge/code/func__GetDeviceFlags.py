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
def _GetDeviceFlags(default_for_blocked_flags=True):
    """Generates the flags for device commands."""
    flags = []
    blocked_state_help_text = 'If blocked, connections from this device will fail.\n\nCan be used to temporarily prevent the device from connecting if, for example, the sensor is generating bad data and needs maintenance.\n\n'
    if not default_for_blocked_flags:
        blocked_state_help_text += '+\n\nUse `--no-blocked` to enable connections and `--blocked` to disable.'
    else:
        blocked_state_help_text += '+\n\nConnections to device is not blocked by default.'
    blocked_default = False if default_for_blocked_flags else None
    flags.append(base.Argument('--blocked', default=blocked_default, action='store_true', help=blocked_state_help_text))
    metadata_key_validator = arg_parsers.RegexpValidator('[a-zA-Z0-9-_]{1,127}', 'Invalid metadata key. Keys should only contain the following characters [a-zA-Z0-9-_] and be fewer than 128 bytes in length.')
    flags.append(base.Argument('--metadata', metavar='KEY=VALUE', type=arg_parsers.ArgDict(key_type=metadata_key_validator), help='The metadata key/value pairs assigned to devices. This metadata is not\ninterpreted or indexed by Cloud IoT Core. It can be used to add contextual\ninformation for the device.\n\nKeys should only contain the following characters ```[a-zA-Z0-9-_]``` and be\nfewer than 128 bytes in length. Values are free-form strings. Each value must\nbe fewer than or equal to 32 KB in size.\n\nThe total size of all keys and values must be less than 256 KB, and the\nmaximum number of key-value pairs is 500.\n'))
    flags.append(base.Argument('--metadata-from-file', metavar='KEY=PATH', type=arg_parsers.ArgDict(key_type=metadata_key_validator), help='Same as --metadata, but the metadata values will be read from the file specified by path.'))
    return flags