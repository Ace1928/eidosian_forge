import contextlib
import os
import sys
from typing import List, Optional, Type, Union
from . import i18n, option, osutils, trace
from .lazy_import import lazy_import
import breezy
from breezy import (
from . import errors, registry
from .hooks import Hooks
from .i18n import gettext
from .plugin import disable_plugins, load_plugins, plugin_name
class HelpCommandIndex:
    """A index for bzr help that returns commands."""

    def __init__(self):
        self.prefix = 'commands/'

    def get_topics(self, topic):
        """Search for topic amongst commands.

        Args:
          topic: A topic to search for.

        Returns:
          A list which is either empty or contains a single
          Command entry.
        """
        if topic and topic.startswith(self.prefix):
            topic = topic[len(self.prefix):]
        try:
            cmd = _get_cmd_object(topic, check_missing=False)
        except KeyError:
            return []
        else:
            return [cmd]