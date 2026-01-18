from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display_info
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core.cache import completion_cache
def AddFlagActionFromAncestors(self, action):
    """Add a flag action to this parser, but segregate it from the others.

    Segregating the action allows automatically generated help text to ignore
    this flag.

    Args:
      action: argparse.Action, The action for the flag being added.
    """
    for flag in ['--project', '--billing-project', '--universe-domain', '--format']:
        if self._FlagArgExists(flag) and flag in action.option_strings:
            return
    self.parser._add_action(action)
    self.data.ancestor_flag_args.append(action)