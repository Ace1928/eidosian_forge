from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def DomainAddAuthorizedNetworksFlag():
    """Defines a flag for adding an authorized network."""
    return base.Argument('--add-authorized-networks', metavar='AUTH_NET1, AUTH_NET2, ...', type=arg_parsers.ArgList(), action=arg_parsers.UpdateAction, help='       A list of URLs of additional networks to peer this domain to in the form\n       projects/{project}/global/networks/{network}.\n       Networks must belong to the project.\n      ')