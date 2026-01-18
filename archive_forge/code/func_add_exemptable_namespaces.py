from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.export import util
from googlecloudsdk.core.console import console_io
def add_exemptable_namespaces(self):
    """Adds handling for configuring exemptable namespaces."""
    group = self.parser.add_argument_group('Exemptable Namespace flags.', mutex=True)
    group.add_argument('--exemptable-namespaces', type=str, help='Namespaces that Policy Controller should ignore, separated by commas if multiple are supplied.')
    group.add_argument('--clear-exemptable-namespaces', action='store_true', help='Removes any namespace exemptions, enabling Policy Controller on all namespaces. Setting this flag will overwrite currently exempted namespaces, not append.')