from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import re
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import display
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import text
import six
def LoadSubElement(self, name, allow_empty=False, release_track_override=None):
    """Load a specific sub group or command.

    Args:
      name: str, The name of the element to load.
      allow_empty: bool, True to allow creating this group as empty to start
        with.
      release_track_override: base.ReleaseTrack, Load the given sub-element
        under the given track instead of that of the parent. This should only
        be used when specifically creating the top level release track groups.

    Returns:
      _CommandCommon, The loaded sub element, or None if it did not exist.
    """
    name = name.replace('-', '_')
    existing = self.groups.get(name, None)
    if not existing:
        existing = self.commands.get(name, None)
    if existing:
        return existing
    if name in self._unloadable_elements:
        return None
    element = None
    try:
        if name in self._groups_to_load:
            element = CommandGroup(self._groups_to_load[name], self._path + [name], release_track_override or self.ReleaseTrack(), self._construction_id, self._cli_generator, self.SubParser(), parent_group=self, allow_empty=allow_empty)
            self.groups[element.name] = element
        elif name in self._commands_to_load:
            element = Command(self._commands_to_load[name], self._path + [name], release_track_override or self.ReleaseTrack(), self._construction_id, self._cli_generator, self.SubParser(), parent_group=self)
            self.commands[element.name] = element
    except command_loading.ReleaseTrackNotImplementedException as e:
        self._unloadable_elements.add(name)
        log.debug(e)
    return element