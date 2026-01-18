from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import enum
import errno
import getpass
import os
import re
import string
import subprocess
import tempfile
import textwrap
from googlecloudsdk.api_lib.oslogin import client as oslogin_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.oslogin import oslogin_utils
from googlecloudsdk.command_lib.util import gaia
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves.urllib.parse import quote
def ParseAndSubstituteSSHFlags(args, remote, instance_address, internal_address):
    """Obtain extra flags from the command arguments."""
    extra_flags = []
    if args.ssh_flag:
        for flag in args.ssh_flag:
            if flag and flag != '--':
                for flag_part in flag.split():
                    deref_flag = flag_part
                    if '%USER%' in deref_flag:
                        deref_flag = deref_flag.replace('%USER%', remote.user)
                    if '%INSTANCE%' in deref_flag:
                        deref_flag = deref_flag.replace('%INSTANCE%', instance_address)
                    if '%INTERNAL%' in deref_flag:
                        deref_flag = deref_flag.replace('%INTERNAL%', internal_address)
                    extra_flags.append(deref_flag)
    return extra_flags