from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAnalyzerShowAccessControlEntries(parser):
    parser.add_argument('--show-response', action='store_true', help='If true, the response will be showed as-is in the command output.')
    parser.set_defaults(show_response=False)