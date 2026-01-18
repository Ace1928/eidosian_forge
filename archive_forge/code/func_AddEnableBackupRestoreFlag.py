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
def AddEnableBackupRestoreFlag(parser):
    """Adds --enable-backup-restore flag to the given parser.

  Args:
    parser: A given parser.
  """
    help_text = '    Enable the Backup for GKE add-on. This add-on is disabled by default. To\n    learn more, see the Backup for GKE overview: https://cloud.google.com/kubernetes-engine/docs/add-on/backup-for-gke/concepts/backup-for-gke.\n    '
    parser.add_argument('--enable-backup-restore', action='store_true', default=None, help=help_text, hidden=False)