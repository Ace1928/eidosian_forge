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
def AddVerifyInternalIpArg(parser):
    parser.add_argument('--verify-internal-ip', action=actions.StoreBooleanProperty(properties.VALUES.ssh.verify_internal_ip), hidden=True, help='Whether or not `gcloud` should perform an initial SSH connection to verify an instance ID is correct when connecting via its internal IP. Without this check, `gcloud` will simply connect to the internal IP of the desired instance, which may be wrong if the desired instance is in a different subnet but happens to share the same internal IP as an instance in the current subnet. Defaults to True.')