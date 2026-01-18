from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
def AddDeprecatedClusterZone(self):
    """Add deprecated cluster zone argument."""
    self.parser.add_argument('--cluster-zone', help='ID of the zone where the cluster is located. Supported zones are listed at https://cloud.google.com/bigtable/docs/locations.', required=False, action=actions.DeprecationAction('--cluster-zone', warn='The {flag_name} argument is deprecated; use --cluster-config instead.', removed=False, action='store'))
    return self