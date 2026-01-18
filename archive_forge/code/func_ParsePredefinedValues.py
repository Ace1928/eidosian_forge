from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import ipaddress
import re
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.privateca import preset_profiles
from googlecloudsdk.command_lib.privateca import text_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from that bucket.
def ParsePredefinedValues(args):
    """Parses an X509Parameters proto message from the predefined values file in args."""
    if not args.IsSpecified('predefined_values_file'):
        return None
    try:
        return messages_util.DictToMessageWithErrorCheck(args.predefined_values_file, privateca_base.GetMessagesModule('v1').X509Parameters)
    except (messages_util.DecodeError, AttributeError):
        raise exceptions.InvalidArgumentException('--predefined-values-file', 'Unrecognized field in the X509Parameters file.')