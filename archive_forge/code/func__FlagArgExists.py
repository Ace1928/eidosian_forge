from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display_info
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core.cache import completion_cache
def _FlagArgExists(self, option_string):
    """If flag with the given option_string exists."""
    for action in self.flag_args:
        if option_string in action.option_strings:
            return True
    return False