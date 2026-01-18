import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
class CommandData:

    def __init__(self, name):
        self.name = name
        self.aliases = [name]
        self.plugin = None
        self.options = []
        self.fixed_words = None