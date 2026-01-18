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
def AddDeviceConfigFlagsToParser(parser):
    """Add flags for the `configs update` command."""
    base.Argument('--version-to-update', type=int, help='          The version number to update. If this value is `0` or unspecified, it\n          will not check the version number of the server and will always update\n          the current version; otherwise, this update will fail if the version\n          number provided does not match the latest version on the server. This\n          is used to detect conflicts with simultaneous updates.\n          ').AddToParser(parser)
    data_group = parser.add_mutually_exclusive_group(required=True)
    base.Argument('--config-file', help='Path to a local file containing the data for this configuration.').AddToParser(data_group)
    base.Argument('--config-data', help='The data for this configuration, as a string. For any values that contain special characters (in the context of your shell), use the `--config-file` flag instead.').AddToParser(data_group)