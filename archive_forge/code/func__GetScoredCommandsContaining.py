from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import re
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _GetScoredCommandsContaining(command_words):
    """Return scored canonical commands containing input command words.

  Args:
    command_words: List of input command words.

  Returns:
    [(canonical_command_words, score)]: List of tuples, where
      canonical_command_words is a list of strings and score is an integer > 0.
      The tuples are sorted from highest score to lowest, and commands with
      the same score appear in lexicographic order.
  """
    root = lookup.LoadCompletionCliTree()
    surface_history = _GetSurfaceHistoryFrequencies(log.GetLogDir())
    normalized_command_words = [command_word.lower().replace('_', '-') for command_word in command_words]
    scored_commands = []
    all_canonical_commands = _GetCanonicalCommands(root)
    canonical_command_set = set(map(tuple, all_canonical_commands))
    for canonical_command_words in all_canonical_commands:
        canonical_command_length = len(canonical_command_words)
        matched = set()
        score = 0
        for index, canonical_command_word in enumerate(canonical_command_words):
            for normalized_command_word in normalized_command_words:
                increment = _WordScore(index, normalized_command_word, canonical_command_word, canonical_command_length)
                alternate_command_word = SYNONYMS.get(normalized_command_word)
                if alternate_command_word:
                    alternate_increment = _WordScore(index, alternate_command_word, canonical_command_word, canonical_command_length)
                    if increment < alternate_increment:
                        increment = alternate_increment
                if increment:
                    matched.add(normalized_command_word)
                    score += increment
        if len(matched) == len(normalized_command_words):
            score += 10
        if score > 0:
            surface = '.'.join(canonical_command_words[:-1])
            if surface in surface_history:
                score += int(surface_history[surface] * FREQUENCY_FACTOR)
            better_track_exists = False
            if 'alpha' == canonical_command_words[0]:
                score -= 5
                if tuple(canonical_command_words[1:]) in canonical_command_set:
                    better_track_exists = True
                if tuple(['beta'] + canonical_command_words[1:]) in canonical_command_set:
                    better_track_exists = True
            if 'beta' == canonical_command_words[0]:
                score -= 5
                if tuple(canonical_command_words[1:]) in canonical_command_set:
                    better_track_exists = True
            if not better_track_exists:
                scored_commands.append((canonical_command_words, score))
    scored_commands.sort(key=lambda tuple: (-tuple[1], tuple[0]))
    return scored_commands