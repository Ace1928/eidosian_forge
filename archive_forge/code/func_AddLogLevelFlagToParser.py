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
def AddLogLevelFlagToParser(parser):
    choices = {'none': 'Disables logging.', 'info': 'Informational events will be logged, such as connections and disconnections. Also includes error events.', 'error': 'Error events will be logged.', 'debug': 'All events will be logged'}
    return base.ChoiceArgument('--log-level', choices=choices, help_str="      The default logging verbosity for activity from devices in this\n        registry. The verbosity level can be overridden by setting a specific\n        device's log level.\n      ").AddToParser(parser)