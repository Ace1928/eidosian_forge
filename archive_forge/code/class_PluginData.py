import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
class PluginData:

    def __init__(self, name, version=None):
        if version is None:
            try:
                version = breezy.plugin.plugins()[name].__version__
            except:
                version = 'unknown'
        self.name = name
        self.version = version

    def __str__(self):
        if self.version == 'unknown':
            return self.name
        return '{} {}'.format(self.name, self.version)