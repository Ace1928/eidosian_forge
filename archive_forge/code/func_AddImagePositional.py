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
def AddImagePositional(parser, verb):
    image_path_format = '*.gcr.io/PROJECT_ID/IMAGE_PATH[:TAG|@sha256:DIGEST]'
    if verb == 'list tags for':
        image_path_format = '*.gcr.io/PROJECT_ID/IMAGE_PATH'
    parser.add_argument('image_name', help='The name of the image to {verb}. The name format should be {image_format}. '.format(verb=verb, image_format=image_path_format))