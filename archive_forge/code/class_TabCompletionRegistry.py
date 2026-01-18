import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
class TabCompletionRegistry:
    """Registry for tab completion responses."""

    def __init__(self):
        self._comp_dict = {}

    def register_tab_comp_context(self, context_words, comp_items):
        """Register a tab-completion context.

    Register that, for each word in context_words, the potential tab-completions
    are the words in comp_items.

    A context word is a pre-existing, completed word in the command line that
    determines how tab-completion works for another, incomplete word in the same
    command line.
    Completion items consist of potential candidates for the incomplete word.

    To give a general example, a context word can be "drink", and the completion
    items can be ["coffee", "tea", "water"]

    Note: A context word can be empty, in which case the context is for the
     top-level commands.

    Args:
      context_words: A list of context words belonging to the context being
        registered. It is a list of str, instead of a single string, to support
        synonym words triggering the same tab-completion context, e.g.,
        both "drink" and the short-hand "dr" can trigger the same context.
      comp_items: A list of completion items, as a list of str.

    Raises:
      TypeError: if the input arguments are not all of the correct types.
    """
        if not isinstance(context_words, list):
            raise TypeError('Incorrect type in context_list: Expected list, got %s' % type(context_words))
        if not isinstance(comp_items, list):
            raise TypeError('Incorrect type in comp_items: Expected list, got %s' % type(comp_items))
        sorted_comp_items = sorted(comp_items)
        for context_word in context_words:
            self._comp_dict[context_word] = sorted_comp_items

    def deregister_context(self, context_words):
        """Deregister a list of context words.

    Args:
      context_words: A list of context words to deregister, as a list of str.

    Raises:
      KeyError: if there are word(s) in context_words that do not correspond
        to any registered contexts.
    """
        for context_word in context_words:
            if context_word not in self._comp_dict:
                raise KeyError('Cannot deregister unregistered context word "%s"' % context_word)
        for context_word in context_words:
            del self._comp_dict[context_word]

    def extend_comp_items(self, context_word, new_comp_items):
        """Add a list of completion items to a completion context.

    Args:
      context_word: A single completion word as a string. The extension will
        also apply to all other context words of the same context.
      new_comp_items: (list of str) New completion items to add.

    Raises:
      KeyError: if the context word has not been registered.
    """
        if context_word not in self._comp_dict:
            raise KeyError('Context word "%s" has not been registered' % context_word)
        self._comp_dict[context_word].extend(new_comp_items)
        self._comp_dict[context_word] = sorted(self._comp_dict[context_word])

    def remove_comp_items(self, context_word, comp_items):
        """Remove a list of completion items from a completion context.

    Args:
      context_word: A single completion word as a string. The removal will
        also apply to all other context words of the same context.
      comp_items: Completion items to remove.

    Raises:
      KeyError: if the context word has not been registered.
    """
        if context_word not in self._comp_dict:
            raise KeyError('Context word "%s" has not been registered' % context_word)
        for item in comp_items:
            self._comp_dict[context_word].remove(item)

    def get_completions(self, context_word, prefix):
        """Get the tab completions given a context word and a prefix.

    Args:
      context_word: The context word.
      prefix: The prefix of the incomplete word.

    Returns:
      (1) None if no registered context matches the context_word.
          A list of str for the matching completion items. Can be an empty list
          of a matching context exists, but no completion item matches the
          prefix.
      (2) Common prefix of all the words in the first return value. If the
          first return value is None, this return value will be None, too. If
          the first return value is not None, i.e., a list, this return value
          will be a str, which can be an empty str if there is no common
          prefix among the items of the list.
    """
        if context_word not in self._comp_dict:
            return (None, None)
        comp_items = self._comp_dict[context_word]
        comp_items = sorted([item for item in comp_items if item.startswith(prefix)])
        return (comp_items, self._common_prefix(comp_items))

    def _common_prefix(self, m):
        """Given a list of str, returns the longest common prefix.

    Args:
      m: (list of str) A list of strings.

    Returns:
      (str) The longest common prefix.
    """
        if not m:
            return ''
        s1 = min(m)
        s2 = max(m)
        for i, c in enumerate(s1):
            if c != s2[i]:
                return s1[:i]
        return s1