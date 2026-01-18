import sys, os
import textwrap
def _populate_option_list(self, option_list, add_help=True):
    if self.standard_option_list:
        self.add_options(self.standard_option_list)
    if option_list:
        self.add_options(option_list)
    if self.version:
        self._add_version_option()
    if add_help:
        self._add_help_option()