from __future__ import absolute_import, division, print_function
import platform
import re
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
class LegacyFactsBase(object):
    COMMANDS = frozenset()

    def __init__(self, module):
        self.module = module
        self.facts = dict()
        self.warnings = list()
        self.responses = None

    def populate(self):
        self.responses = run_commands(self.module, list(self.COMMANDS))