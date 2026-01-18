from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os.path
import threading
import time
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.queued_resources import ssh as qr_ssh_utils
from googlecloudsdk.command_lib.compute.tpus.queued_resources import util as queued_resource_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import ssh as tpu_ssh_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def AddCommandArgGroup(parser):
    """Argument group for running commands using SSH."""
    command_group = parser.add_argument_group(help='These arguments are used to run commands using SSH.')
    command_group.add_argument('--command', required=True, help="      Command to run on the Cloud TPU VM.\n\n      Runs the command on the target Cloud TPU Queued Resource's nodes and then exits.\n\n      Note: in the case of a TPU Pod, it will only run the command in the\n      workers specified with the `--worker` flag (defaults to worker all if not\n      set).\n      ")
    command_group.add_argument('--output-directory', help='      Path to the directory to output the logs of the commands.\n\n      The path can be relative or absolute. The directory must already exist.\n\n      If not specified, standard output will be used.\n\n      The logs will be written in files named {WORKER_ID}.log. For example:\n      "2.log".\n      ')