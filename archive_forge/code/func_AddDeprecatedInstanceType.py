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
def AddDeprecatedInstanceType(self):
    """Add deprecated instance type argument."""
    choices = {'PRODUCTION': 'Production instances provide high availability and are suitable for applications in production. Production instances created with the --instance-type argument have 3 nodes if a value is not provided for --cluster-num-nodes.', 'DEVELOPMENT': 'Development instances are low-cost instances meant for development and testing only. They do not provide high availability and no service level agreement applies.'}
    self.parser.add_argument('--instance-type', default='PRODUCTION', type=lambda x: x.upper(), choices=choices, help='The type of instance to create.', required=False, action=actions.DeprecationAction('--instance-type', warn='The {flag_name} argument is deprecated. DEVELOPMENT instances are no longer offered. All instances are of type PRODUCTION.', removed=False, action='store'))
    return self