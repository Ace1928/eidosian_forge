from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import re
import textwrap
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def AddCommonArgs(parser):
    """Adds the flags shared between 'hub' subcommands to parser.

  Args:
    parser: an argparse.ArgumentParser, to which the common flags will be added
  """
    parser.add_argument('--kubeconfig', type=str, help=textwrap.dedent('          The kubeconfig file containing an entry for the cluster. Defaults to\n          $KUBECONFIG if it is set in the environment, otherwise defaults to\n          to $HOME/.kube/config.\n        '))
    parser.add_argument('--context', type=str, help=textwrap.dedent('        The context in the kubeconfig file that specifies the cluster.\n      '))