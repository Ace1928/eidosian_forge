from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.core.document_renderers import render_document
import six
from six.moves import filter
def LocateTerm(command, term):
    """Helper function to get first location of term in a json command.

  Locations are considered in this order: 'name', 'capsule',
  'sections', 'positionals', 'flags', 'commands', 'path'. Returns a dot-
  separated lookup for the location e.g. 'sections.description' or
  empty string if not found.

  Args:
    command: dict, json representation of command.
    term: str, the term to search.

  Returns:
    str, lookup for where to find the term when building summary of command.
  """
    if command[lookup.IS_HIDDEN]:
        return ''
    regexp = re.compile(re.escape(term), re.IGNORECASE)
    if regexp.search(command[lookup.NAME]):
        return lookup.NAME
    if regexp.search(' '.join(command[lookup.PATH] + [lookup.NAME])):
        return lookup.PATH

    def _Flags(command):
        return {flag_name: flag for flag_name, flag in six.iteritems(command[lookup.FLAGS]) if not flag[lookup.IS_HIDDEN] and (not flag[lookup.IS_GLOBAL])}
    for flag_name, flag in sorted(six.iteritems(_Flags(command))):
        if regexp.search(flag_name):
            return DOT.join([lookup.FLAGS, flag[lookup.NAME], lookup.NAME])
    for positional in command[lookup.POSITIONALS]:
        if regexp.search(positional[lookup.NAME]):
            return DOT.join([lookup.POSITIONALS, positional[lookup.NAME], lookup.NAME])
    if regexp.search(command[lookup.CAPSULE]):
        return lookup.CAPSULE
    for section_name, section_desc in sorted(six.iteritems(command[lookup.SECTIONS])):
        if regexp.search(section_desc):
            return DOT.join([lookup.SECTIONS, section_name])
    for flag_name, flag in sorted(six.iteritems(_Flags(command))):
        for sub_attribute in [lookup.CHOICES, lookup.DESCRIPTION, lookup.DEFAULT]:
            if regexp.search(six.text_type(flag.get(sub_attribute, ''))):
                return DOT.join([lookup.FLAGS, flag[lookup.NAME], sub_attribute])
    for positional in command[lookup.POSITIONALS]:
        if regexp.search(positional[lookup.DESCRIPTION]):
            return DOT.join([lookup.POSITIONALS, positional[lookup.NAME], positional[lookup.DESCRIPTION]])
    if regexp.search(six.text_type([n for n, c in six.iteritems(command[lookup.COMMANDS]) if not c[lookup.IS_HIDDEN]])):
        return lookup.COMMANDS
    return ''