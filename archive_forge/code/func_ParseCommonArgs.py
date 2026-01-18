from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
def ParseCommonArgs(parser):
    """Parses arguments common to all cloud-shell commands."""
    parser.add_argument('--force-key-file-overwrite', help='      If enabled gcloud will regenerate and overwrite the files associated\n      with a broken SSH key without asking for confirmation in both\n      interactive and non-interactive environment.\n\n      If disabled gcloud will not attempt to regenerate the files associated\n      with a broken SSH key and fail in both interactive and non-interactive\n      environment.\n      ', action='store_true')
    parser.add_argument('--ssh-key-file', help='      The path to the SSH key file. By default, this is\n        *~/.ssh/google_compute_engine*.\n      ', action='store_true')