from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import difflib
import enum
import io
import re
import sys
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import util as format_util
import six
class TextChoiceSuggester(object):
    """Utility to suggest mistyped commands.

  """

    def __init__(self, choices=None):
        self._choices = {}
        if choices:
            self.AddChoices(choices)

    def AddChoices(self, choices):
        """Add a set of valid things that can be suggested.

    Args:
      choices: [str], The valid choices.
    """
        for choice in choices:
            if choice not in self._choices:
                self._choices[choice] = choice

    def AddAliases(self, aliases, suggestion):
        """Add an alias that is not actually a valid choice, but will suggest one.

    This should be called after AddChoices() so that aliases will not clobber
    any actual choices.

    Args:
      aliases: [str], The aliases for the valid choice.  This is something
        someone will commonly type when they actually mean something else.
      suggestion: str, The valid choice to suggest.
    """
        for alias in aliases:
            if alias not in self._choices:
                self._choices[alias] = suggestion

    def GetSuggestion(self, arg):
        """Find the item that is closest to what was attempted.

    Args:
      arg: str, The argument provided.

    Returns:
      str, The closest match.
    """
        if not self._choices:
            return None
        match = difflib.get_close_matches(arg.lower(), [six.text_type(c) for c in self._choices], 1)
        if match:
            choice = [c for c in self._choices if six.text_type(c) == match[0]][0]
            return self._choices[choice]
        return self._choices[match[0]] if match else None