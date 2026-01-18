from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib import tasks as tasks_api_lib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddPayloadFlags(parser, is_alpha=False):
    """Adds either payload or body flags."""
    payload_group = parser.add_mutually_exclusive_group()
    if is_alpha:
        payload_group.add_argument('--payload-content', help='            Data payload used by the task worker to process the task.\n            ')
        payload_group.add_argument('--payload-file', help='            File containing data payload used by the task worker to process the\n            task.\n            ')
    else:
        payload_group.add_argument('--body-content', help='            HTTP Body data sent to the task worker processing the task.\n            ')
        payload_group.add_argument('--body-file', help='            File containing HTTP body data sent to the task worker processing\n            the task.\n            ')