from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def DomainRemoveAuthorizedNetworksFlag():
    """Defines a flag for removing an authorized network."""
    return base.Argument('--remove-authorized-networks', metavar='AUTH_NET1, AUTH_NET2, ...', type=arg_parsers.ArgList(), action=arg_parsers.UpdateAction, help='       A list of URLs of additional networks to unpeer this domain from in the\n       form projects/{project}/global/networks/{network}.\n       Networks must belong to the project.\n      ')