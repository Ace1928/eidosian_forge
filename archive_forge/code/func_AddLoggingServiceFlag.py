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
def AddLoggingServiceFlag(parser):
    """Adds a --logging-service flag to the parser.

  Args:
    parser: A given parser.
  """
    help_str = 'Logging service to use for the cluster. Options are:\n"logging.googleapis.com/kubernetes" (the Google Cloud Logging\nservice with Kubernetes-native resource model enabled),\n"logging.googleapis.com" (the Google Cloud Logging service),\n"none" (logs will not be exported from the cluster)\n'
    parser.add_argument('--logging-service', action=actions.DeprecationAction('--logging-service', warn='The `--logging-service` flag is deprecated and will be removed in an upcoming release. Please use `--logging` instead. For more information, please read: https://cloud.google.com/stackdriver/docs/solutions/gke/installing.'), help=help_str)