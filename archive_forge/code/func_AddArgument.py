from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.dataplex import parsers as dataplex_parsers
from googlecloudsdk.command_lib.util.args import labels_util
def AddArgument(name: str, **kwargs):
    parser_to_add = entry_source
    if for_update:
        parser_to_add = entry_source.add_mutually_exclusive_group()
        parser_to_add.add_argument('--clear-entry-source-' + name, action='store_true', help=f'Clear the value for the {name.replace('-', '_')} field in the Entry Source.')
    parser_to_add.add_argument('--entry-source-' + name, **kwargs)