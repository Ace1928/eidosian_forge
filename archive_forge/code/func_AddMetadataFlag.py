from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddMetadataFlag(parser, resource_type, dictionary_size_limit):
    """Adds metadata flags for service-directory commands."""
    return base.Argument('--metadata', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help='           Metadata for the {}.\n\n           Metadata takes the form of key/value string pairs. Keys are\n           composed of an optional prefix and a name segment, separated by a\n           slash(/). Prefixes and names must be composed of alphanumeric\n           characters, dashes, and dots. Names may also use underscores. There\n           are no character restrictions on what may go into the value of a\n           metadata. The entire dictionary is limited to {} characters, spread\n           across all key-value pairs.\n           '.format(resource_type, dictionary_size_limit)).AddToParser(parser)