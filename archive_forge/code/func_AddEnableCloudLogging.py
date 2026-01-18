from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddEnableCloudLogging(parser):
    parser.add_argument('--enable-cloud-logging', action=actions.DeprecationAction('--enable-cloud-logging', show_message=lambda val: val, warn='Legacy Logging and Monitoring is deprecated. Thus, flag `--enable-cloud-logging` is also deprecated and will be removed in an upcoming release. Please use `--logging` (optionally with `--monitoring`). For more details, please read: https://cloud.google.com/stackdriver/docs/solutions/gke/installing.', action='store_true'), help='Automatically send logs from the cluster to the Google Cloud Logging API.')