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
def AddImageTypeFlag(parser, target):
    """Adds a --image-type flag to the given parser."""
    help_text = 'The image type to use for the {target}. Defaults to server-specified.\n\nImage Type specifies the base OS that the nodes in the {target} will run on.\nIf an image type is specified, that will be assigned to the {target} and all\nfuture upgrades will use the specified image type. If it is not specified the\nserver will pick the default image type.\n\nThe default image type and the list of valid image types are available\nusing the following command.\n\n  $ gcloud container get-server-config\n'.format(target=target)
    parser.add_argument('--image-type', help=help_text)