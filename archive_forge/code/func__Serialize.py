from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def _Serialize(tree):
    """Returns the CLI tree optimized for serialization.

  Serialized data does not support pointers. The CLI tree can have a lot of
  redundant data, especially with ancestor flags included with each command.
  This function collects the flags into the _LOOKUP_SERIALIZED_FLAG_LIST array
  in the root node and converts the flags dict values to indices into that
  array.

  Serialization saves a lot of space and allows the ancestor flags to be
  included in the LOOKUP_FLAGS dict of each command. It also saves time for
  users of the tree because the LOOKUP_FLAGS dict also contains the ancestor
  flags.

  Apply this function to the CLI tree just before dumping. For the 2017-03
  gcloud CLI with alpha and beta included and all ancestor flags included in
  each command node this function reduces the generation time from
  ~2m40s to ~35s and the dump file size from 35Mi to 4.3Mi.

  Args:
    tree: The CLI tree to be optimized.

  Returns:
    The CLI tree optimized for serialization.
  """
    if getattr(tree, _LOOKUP_SERIALIZED_FLAG_LIST, None):
        return tree
    all_flags = {}

    class _FlagIndex(object):
        """Flag index + definition."""

        def __init__(self, flag):
            self.flag = flag
            self.index = 0

    def _FlagIndexKey(flag):
        return '::'.join([six.text_type(flag.name), six.text_type(flag.attr), six.text_type(flag.category), '[{}]'.format(', '.join((six.text_type(c) for c in flag.choices))), six.text_type(flag.completer), six.text_type(flag.default), six.text_type(flag.description), six.text_type(flag.is_hidden), six.text_type(flag.is_global), six.text_type(flag.is_group), six.text_type(flag.is_required), six.text_type(flag.nargs), six.text_type(flag.type), six.text_type(flag.value)])

    def _CollectAllFlags(command):
        for flag in command.flags.values():
            all_flags[_FlagIndexKey(flag)] = _FlagIndex(flag)
        for subcommand in command.commands.values():
            _CollectAllFlags(subcommand)
    _CollectAllFlags(tree)
    all_flags_list = []
    for index, key in enumerate(sorted(all_flags)):
        fi = all_flags[key]
        fi.index = index
        all_flags_list.append(fi.flag)

    def _ReplaceConstraintFlagWithIndex(arguments):
        positional_index = 0
        for i, arg in enumerate(arguments):
            if isinstance(arg, int):
                pass
            elif arg.is_group:
                _ReplaceConstraintFlagWithIndex(arg.arguments)
            elif arg.is_positional:
                positional_index -= 1
                arguments[i] = positional_index
            else:
                try:
                    arguments[i] = all_flags[_FlagIndexKey(arg)].index
                except KeyError:
                    pass

    def _ReplaceFlagWithIndex(command):
        for name, flag in six.iteritems(command.flags):
            command.flags[name] = all_flags[_FlagIndexKey(flag)].index
            _ReplaceConstraintFlagWithIndex(command.constraints.arguments)
        for subcommand in command.commands.values():
            _ReplaceFlagWithIndex(subcommand)
    _ReplaceFlagWithIndex(tree)
    setattr(tree, _LOOKUP_SERIALIZED_FLAG_LIST, all_flags_list)
    return tree