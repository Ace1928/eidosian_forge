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
def GetSSHKeyExpirationFromArgs(args):
    """Converts flags to an ssh key expiration in datetime and micros."""
    if args.ssh_key_expiration:
        expiration = args.ssh_key_expiration
    elif args.ssh_key_expire_after:
        expiration = times.Now() + datetime.timedelta(seconds=args.ssh_key_expire_after)
    else:
        return (None, None)
    expiration_micros = times.GetTimeStampFromDateTime(expiration) * 1000000.0
    return (expiration, int(expiration_micros))