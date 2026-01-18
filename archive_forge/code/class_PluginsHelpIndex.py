import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
class PluginsHelpIndex:
    """A help index that returns help topics for plugins."""

    def __init__(self):
        self.prefix = 'plugins/'

    def get_topics(self, topic):
        """Search for topic in the loaded plugins.

        This will not trigger loading of new plugins.

        Args:
          topic: A topic to search for.

        Returns:
          A list which is either empty or contains a single
          RegisteredTopic entry.
        """
        if not topic:
            return []
        if topic.startswith(self.prefix):
            topic = topic[len(self.prefix):]
        plugin_module_name = _MODULE_PREFIX + topic
        try:
            module = sys.modules[plugin_module_name]
        except KeyError:
            return []
        else:
            return [ModuleHelpTopic(module)]