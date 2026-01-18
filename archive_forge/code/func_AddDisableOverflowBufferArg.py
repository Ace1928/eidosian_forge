from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def AddDisableOverflowBufferArg(parser):
    """Adds an argument for determining whether to backlog the execution."""
    parser.add_argument('--disable-concurrency-quota-overflow-buffering', action='store_true', default=False, help='If set, the execution will not be backlogged when the concurrency quota is exhausted. Backlogged executions start when the concurrency quota becomes available.')