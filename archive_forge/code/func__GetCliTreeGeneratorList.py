from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.meta import generate_cli_trees
def _GetCliTreeGeneratorList():
    return ', '.join(sorted([cli_tree.DEFAULT_CLI_NAME] + list(generate_cli_trees.GENERATORS.keys())))