from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import collections
import datetime
import json
import os
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
from googlecloudsdk.core.util.files import FileReader
from googlecloudsdk.core.util.files import FileWriter
import six
def AddSSHKeyExpirationArgs(parser):
    """Additional flags to handle expiring SSH keys."""
    group = parser.add_mutually_exclusive_group()

    def ParseFutureDatetime(s):
        """Parses a string value into a future Datetime object."""
        dt = arg_parsers.Datetime.Parse(s)
        if dt < times.Now():
            raise arg_parsers.ArgumentTypeError('Date/time must be in the future: {0}'.format(s))
        return dt
    group.add_argument('--ssh-key-expiration', type=ParseFutureDatetime, help='        The time when the ssh key will be valid until, such as\n        "2017-08-29T18:52:51.142Z." This is only valid if the instance is not\n        using OS Login. See $ gcloud topic datetimes for information on time\n        formats.\n        ')
    group.add_argument('--ssh-key-expire-after', type=arg_parsers.Duration(lower_bound='1s'), help='        The maximum length of time an SSH key is valid for once created and\n        installed, e.g. 2m for 2 minutes. See $ gcloud topic datetimes for\n        information on duration formats.\n      ')