from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def AddAuthorizedOrgsDescUpdateArgs(parser):
    repeated.AddPrimitiveArgs(parser, 'authorized_orgs_desc', 'orgs', 'orgs', additional_help='Orgs must be organizations, in the form `organizations/<organizationsnumber>`.')